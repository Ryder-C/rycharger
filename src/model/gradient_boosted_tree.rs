use serde::{Deserialize, Serialize};

use super::{
    ChargeModel, Features, Prediction, Session, NUM_FEATURES, avg_session_length,
    session_to_example, sigmoid,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Node {
    Split {
        feature_index: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
    Leaf {
        weight: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn predict(&self, features: &Features) -> f64 {
        let mut idx = 0;
        loop {
            match &self.nodes[idx] {
                Node::Leaf { weight } => return *weight,
                Node::Split {
                    feature_index,
                    threshold,
                    left,
                    right,
                } => {
                    idx = if features.values[*feature_index] <= *threshold {
                        *left
                    } else {
                        *right
                    };
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientBoostedTree {
    trees: Vec<Tree>,
    base_score: f64,
    feature_importance: Vec<f64>,
    trained_count: usize,
    running_session_total_mins: f64,
    running_session_count: usize,
    // Hyperparameters
    max_trees: usize,
    max_depth: u32,
    learning_rate: f64,
    lambda: f64,
    min_samples_split: usize,
}

impl Default for GradientBoostedTree {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientBoostedTree {
    pub fn new() -> Self {
        Self {
            trees: Vec::new(),
            base_score: 0.0,
            feature_importance: vec![0.0; NUM_FEATURES],
            trained_count: 0,
            running_session_total_mins: 0.0,
            running_session_count: 0,
            max_trees: 50,
            max_depth: 4,
            learning_rate: 0.1,
            lambda: 1.0,
            min_samples_split: 3,
        }
    }

    pub fn feature_importance(&self) -> &[f64] {
        &self.feature_importance
    }

    fn raw_score(&self, features: &Features) -> f64 {
        let tree_sum: f64 = self.trees.iter().map(|t| t.predict(features)).sum();
        self.base_score + self.learning_rate * tree_sum
    }

    fn build_tree(
        &mut self,
        examples: &[(Features, f64)],
        gradients: &[f64],
        hessians: &[f64],
        indices: &[usize],
        depth: u32,
        nodes: &mut Vec<Node>,
    ) -> usize {
        let node_idx = nodes.len();

        // Leaf conditions
        if depth >= self.max_depth || indices.len() < self.min_samples_split {
            let g_sum: f64 = indices.iter().map(|&i| gradients[i]).sum();
            let h_sum: f64 = indices.iter().map(|&i| hessians[i]).sum();
            let weight = -g_sum / (h_sum + self.lambda);
            nodes.push(Node::Leaf { weight });
            return node_idx;
        }

        // Find best split
        let mut best_gain = 0.0_f64;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_left_indices = Vec::new();
        let mut best_right_indices = Vec::new();

        let g_total: f64 = indices.iter().map(|&i| gradients[i]).sum();
        let h_total: f64 = indices.iter().map(|&i| hessians[i]).sum();

        let mut sorted_indices: Vec<usize> = indices.to_vec();

        for feature in 0..NUM_FEATURES {
            sorted_indices.sort_by(|&a, &b| {
                examples[a].0.values[feature]
                    .partial_cmp(&examples[b].0.values[feature])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut g_left = 0.0;
            let mut h_left = 0.0;

            for split_pos in 0..sorted_indices.len() - 1 {
                let i = sorted_indices[split_pos];
                g_left += gradients[i];
                h_left += hessians[i];

                // Skip if next sample has same feature value
                let next_i = sorted_indices[split_pos + 1];
                if (examples[i].0.values[feature] - examples[next_i].0.values[feature]).abs()
                    < 1e-12
                {
                    continue;
                }

                let g_right = g_total - g_left;
                let h_right = h_total - h_left;

                let gain = 0.5
                    * (g_left.powi(2) / (h_left + self.lambda)
                        + g_right.powi(2) / (h_right + self.lambda)
                        - g_total.powi(2) / (h_total + self.lambda));

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature;
                    best_threshold =
                        (examples[i].0.values[feature] + examples[next_i].0.values[feature]) / 2.0;
                    best_left_indices = sorted_indices[..=split_pos].to_vec();
                    best_right_indices = sorted_indices[split_pos + 1..].to_vec();
                }
            }
        }

        if best_gain <= 0.0 {
            let weight = -g_total / (h_total + self.lambda);
            nodes.push(Node::Leaf { weight });
            return node_idx;
        }

        // Record feature importance
        self.feature_importance[best_feature] += best_gain;

        // Reserve slot for split node
        nodes.push(Node::Leaf { weight: 0.0 }); // placeholder

        let left = self.build_tree(
            examples,
            gradients,
            hessians,
            &best_left_indices,
            depth + 1,
            nodes,
        );
        let right = self.build_tree(
            examples,
            gradients,
            hessians,
            &best_right_indices,
            depth + 1,
            nodes,
        );

        nodes[node_idx] = Node::Split {
            feature_index: best_feature,
            threshold: best_threshold,
            left,
            right,
        };

        node_idx
    }
}

impl ChargeModel for GradientBoostedTree {
    fn predict(&self, features: &Features, threshold: f64) -> Prediction {
        let prob = sigmoid(self.raw_score(features));
        Prediction {
            unplug_probability: prob,
            should_charge_to_full: prob >= threshold,
        }
    }

    fn train(&mut self, sessions: &[Session], horizon_mins: u64) {
        if sessions.is_empty() {
            return;
        }

        let avg_len = avg_session_length(sessions);

        let examples: Vec<(Features, f64)> = sessions
            .iter()
            .map(|s| session_to_example(s, horizon_mins, avg_len))
            .collect();

        // Compute base score from class prior
        let pos_count = examples.iter().filter(|(_, y)| *y > 0.5).count() as f64;
        let p = (pos_count / examples.len() as f64).clamp(0.01, 0.99);
        self.base_score = (p / (1.0 - p)).ln();

        self.trees.clear();
        self.feature_importance = vec![0.0; NUM_FEATURES];

        let n = examples.len();

        for _ in 0..self.max_trees {
            // Compute predictions, gradients, hessians
            let mut gradients = Vec::with_capacity(n);
            let mut hessians = Vec::with_capacity(n);
            let mut max_abs_grad = 0.0_f64;

            for (features, label) in &examples {
                let pred = sigmoid(self.raw_score(features));
                let g = pred - *label;
                let h = pred * (1.0 - pred);
                max_abs_grad = max_abs_grad.max(g.abs());
                gradients.push(g);
                hessians.push(h);
            }

            // Early stopping
            if max_abs_grad < 1e-6 {
                break;
            }

            let indices: Vec<usize> = (0..n).collect();
            let mut nodes = Vec::new();
            self.build_tree(&examples, &gradients, &hessians, &indices, 0, &mut nodes);
            self.trees.push(Tree { nodes });
        }

        self.trained_count = sessions.len();
        self.running_session_total_mins = avg_len * sessions.len() as f64;
        self.running_session_count = sessions.len();
    }

    fn update(&mut self, session: &Session, horizon_mins: u64) {
        let duration_mins = session
            .unplugged_at
            .signed_duration_since(session.plugged_in_at)
            .num_minutes() as f64;

        self.running_session_total_mins += duration_mins;
        self.running_session_count += 1;
        let avg = self.running_session_total_mins / self.running_session_count as f64;

        if self.trees.len() < self.max_trees {
            let (features, label) = session_to_example(session, horizon_mins, avg);
            let pred = sigmoid(self.raw_score(&features));
            let g = pred - label;
            let h = pred * (1.0 - pred);
            let weight = -g / (h + self.lambda);

            self.trees.push(Tree {
                nodes: vec![Node::Leaf { weight }],
            });
        }

        self.trained_count += 1;
    }

    fn training_count(&self) -> usize {
        self.trained_count
    }

    fn avg_session_length_mins(&self) -> f64 {
        if self.running_session_count == 0 {
            return 0.0;
        }
        self.running_session_total_mins / self.running_session_count as f64
    }
}
