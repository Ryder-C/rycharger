use serde::{Deserialize, Serialize};

use super::{
    ChargeModel, Features, NUM_FEATURES, Prediction, RunningStats, Session, session_to_example,
    sigmoid,
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

struct SplitResult {
    gain: f64,
    feature_index: usize,
    threshold: f64,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
}

/// Find the best feature/threshold split for the given subset of examples.
fn find_best_split(
    examples: &[(Features, f64)],
    gradients: &[f64],
    hessians: &[f64],
    indices: &[usize],
    lambda: f64,
) -> Option<SplitResult> {
    let g_total: f64 = indices.iter().map(|&i| gradients[i]).sum();
    let h_total: f64 = indices.iter().map(|&i| hessians[i]).sum();

    let mut best_gain = 0.0_f64;
    let mut best: Option<SplitResult> = None;
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
            if (examples[i].0.values[feature] - examples[next_i].0.values[feature]).abs() < 1e-12
            {
                continue;
            }

            let g_right = g_total - g_left;
            let h_right = h_total - h_left;

            let gain = 0.5
                * (g_left.powi(2) / (h_left + lambda)
                    + g_right.powi(2) / (h_right + lambda)
                    - g_total.powi(2) / (h_total + lambda));

            if gain > best_gain {
                best_gain = gain;
                best = Some(SplitResult {
                    gain,
                    feature_index: feature,
                    threshold: (examples[i].0.values[feature]
                        + examples[next_i].0.values[feature])
                        / 2.0,
                    left_indices: sorted_indices[..=split_pos].to_vec(),
                    right_indices: sorted_indices[split_pos + 1..].to_vec(),
                });
            }
        }
    }

    best
}

/// Compute the optimal leaf weight for a set of sample indices.
fn leaf_weight(gradients: &[f64], hessians: &[f64], indices: &[usize], lambda: f64) -> f64 {
    let g_sum: f64 = indices.iter().map(|&i| gradients[i]).sum();
    let h_sum: f64 = indices.iter().map(|&i| hessians[i]).sum();
    -g_sum / (h_sum + lambda)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientBoostedTree {
    trees: Vec<Tree>,
    base_score: f64,
    feature_importance: Vec<f64>,
    trained_count: usize,
    stats: RunningStats,
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
            stats: RunningStats::new(),
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

        if depth >= self.max_depth || indices.len() < self.min_samples_split {
            let weight = leaf_weight(gradients, hessians, indices, self.lambda);
            nodes.push(Node::Leaf { weight });
            return node_idx;
        }

        let split = match find_best_split(examples, gradients, hessians, indices, self.lambda) {
            Some(s) => s,
            None => {
                let weight = leaf_weight(gradients, hessians, indices, self.lambda);
                nodes.push(Node::Leaf { weight });
                return node_idx;
            }
        };

        self.feature_importance[split.feature_index] += split.gain;

        // Reserve slot for split node (replaced after children are built)
        nodes.push(Node::Leaf { weight: 0.0 });

        let left = self.build_tree(
            examples,
            gradients,
            hessians,
            &split.left_indices,
            depth + 1,
            nodes,
        );
        let right = self.build_tree(
            examples,
            gradients,
            hessians,
            &split.right_indices,
            depth + 1,
            nodes,
        );

        nodes[node_idx] = Node::Split {
            feature_index: split.feature_index,
            threshold: split.threshold,
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

        self.stats = RunningStats::from_sessions(sessions);
        let avg_len = self.stats.average();

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

            if max_abs_grad < 1e-6 {
                break;
            }

            let indices: Vec<usize> = (0..n).collect();
            let mut nodes = Vec::new();
            self.build_tree(&examples, &gradients, &hessians, &indices, 0, &mut nodes);
            self.trees.push(Tree { nodes });
        }

        self.trained_count = sessions.len();
    }

    fn update(&mut self, session: &Session, horizon_mins: u64) {
        self.stats.update(session);
        let avg = self.stats.average();

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
        self.stats.average()
    }
}
