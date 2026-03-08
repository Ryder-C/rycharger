mod gradient_boosted_tree;
mod logistic_regression;

pub use gradient_boosted_tree::GradientBoostedTree;
pub use logistic_regression::LogisticRegression;

use chrono::{Datelike, NaiveDateTime, Timelike};
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

/// A recorded charging session
pub struct Session {
    pub plugged_in_at: NaiveDateTime,
    pub unplugged_at: NaiveDateTime,
}

/// Extracted features
pub struct Features {
    pub values: Vec<f64>,
}

pub struct Prediction {
    /// Probability that the user will unplug within prediction horizon
    pub unplug_probability: f64,

    /// Whether we should charge to full based on threshold
    pub should_charge_to_full: bool,
}

/// Tracks running average of session durations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RunningStats {
    total_mins: f64,
    count: usize,
}

impl RunningStats {
    pub fn new() -> Self {
        Self {
            total_mins: 0.0,
            count: 0,
        }
    }

    pub fn from_sessions(sessions: &[Session]) -> Self {
        let total_mins: f64 = sessions
            .iter()
            .map(|s| {
                s.unplugged_at
                    .signed_duration_since(s.plugged_in_at)
                    .num_minutes() as f64
            })
            .sum();
        Self {
            total_mins,
            count: sessions.len(),
        }
    }

    pub fn update(&mut self, session: &Session) {
        let duration_mins = session
            .unplugged_at
            .signed_duration_since(session.plugged_in_at)
            .num_minutes() as f64;
        self.total_mins += duration_mins;
        self.count += 1;
    }

    pub fn average(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_mins / self.count as f64
        }
    }
}

pub trait ChargeModel {
    /// Predict whether the user will need a full charge soon
    fn predict(&self, features: &Features, threshold: f64) -> Prediction;

    /// Train on batch of session data
    fn train(&mut self, sessions: &[Session], horizon_mins: u64);

    /// Incrementally update with a single session
    fn update(&mut self, session: &Session, horizon_mins: u64);

    /// Number of sessions used for training
    fn training_count(&self) -> usize;

    /// Average session length from running stats
    fn avg_session_length_mins(&self) -> f64;
}

pub(crate) const NUM_FEATURES: usize = 11; // 2 (hour) + 7 (day) + 1 (plugged duration) + 1 (avg session)

pub(crate) fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

pub(crate) fn session_to_example(
    session: &Session,
    horizon_mins: u64,
    avg_session_len: f64,
) -> (Features, f64) {
    let duration_mins = session
        .unplugged_at
        .signed_duration_since(session.plugged_in_at)
        .num_minutes() as f64;

    let features = Features::extract(session.plugged_in_at, 0.0, avg_session_len);
    let label = if duration_mins <= horizon_mins as f64 {
        1.0
    } else {
        0.0
    };

    (features, label)
}

impl Features {
    /// Extract features from the current moment + context
    pub fn extract(
        now: NaiveDateTime,
        minutes_plugged_in: f64,
        avg_session_length_mins: f64,
    ) -> Self {
        let mut values = Vec::with_capacity(NUM_FEATURES);

        // Hour encoding (cyclic)
        let hour_frac = now.hour() as f64 + now.minute() as f64 / 60.0;
        values.push((hour_frac * TAU / 24.0).sin());
        values.push((hour_frac * TAU / 24.0).cos());

        // Day of week encoding (one-hot)
        let dow = now.weekday().number_from_monday() as usize - 1;
        for i in 0..7 {
            values.push(if i == dow { 1.0 } else { 0.0 });
        }

        // Normalized plugged in duration (cap at 8 hours)
        values.push((minutes_plugged_in / 480.0).min(1.0));
        values.push((avg_session_length_mins / 480.0).min(1.0));

        Self { values }
    }
}
