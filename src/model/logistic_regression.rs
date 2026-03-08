use serde::{Deserialize, Serialize};

use super::{
    ChargeModel, Features, NUM_FEATURES, Prediction, Session, avg_session_length,
    session_to_example, sigmoid,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegression {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    trained_count: usize,

    running_session_total_mins: f64,
    running_session_count: usize,
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
            running_session_total_mins: 0.0,
            running_session_count: 0,
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
        let avg_len = avg_session_length(sessions);

        for _ in 0..10 {
            for session in sessions {
                let (features, label) = session_to_example(session, horizon_mins, avg_len);
                self.sgd_step(&features, label);
            }
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

        let (features, label) = session_to_example(session, horizon_mins, avg);
        self.sgd_step(&features, label);
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
