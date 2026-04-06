use serde::{Deserialize, Serialize};

use super::{ChargeModel, Features, NUM_FEATURES, Prediction, RunningStats, Session, session_to_examples, sigmoid};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegression {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    trained_count: usize,
    stats: RunningStats,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl LogisticRegression {
    pub fn new() -> Self {
        Self {
            weights: vec![0.0; NUM_FEATURES],
            bias: 0.0,
            learning_rate: 0.01,
            trained_count: 0,
            stats: RunningStats::new(),
        }
    }

    fn forward(&self, features: &Features) -> f64 {
        let z: f64 = self
            .weights
            .iter()
            .zip(&features.values)
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.bias;

        sigmoid(z)
    }

    fn sgd_step(&mut self, features: &Features, label: f64) {
        let pred = self.forward(features);
        let error = pred - label;

        for (w, x) in self.weights.iter_mut().zip(&features.values) {
            *w -= self.learning_rate * error * x;
        }
        self.bias -= self.learning_rate * error;
    }
}

impl ChargeModel for LogisticRegression {
    fn predict(&self, features: &Features, threshold: f64) -> Prediction {
        let prob = self.forward(features);
        Prediction {
            unplug_probability: prob,
            should_charge_to_full: prob >= threshold,
        }
    }

    fn train(&mut self, sessions: &[Session], horizon_mins: u64) {
        self.stats = RunningStats::from_sessions(sessions);
        let avg_len = self.stats.average();

        for _ in 0..10 {
            for session in sessions {
                let examples = session_to_examples(session, horizon_mins, avg_len);
                for (features, label) in examples {
                    self.sgd_step(&features, label);
                }
            }
        }
        self.trained_count = sessions.len();
    }

    fn update(&mut self, session: &Session, horizon_mins: u64) {
        self.stats.update(session);
        let avg = self.stats.average();

        let examples = session_to_examples(session, horizon_mins, avg);
        for (features, label) in examples {
            self.sgd_step(&features, label);
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
