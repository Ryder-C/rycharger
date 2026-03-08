use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{Local, NaiveDateTime};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::collection::{self, Event};
use crate::config::Config;
use crate::model::{ChargeModel, Features, LogisticRegression};
use crate::storage::{Database, ModelWeights};

pub async fn run(config: Arc<Config>) -> Result<()> {
    let db = Database::open(&config.daemon.db_path)?;
    let mut model = load_or_train_model(&db, &config)?;

    let (tx, mut rx) = mpsc::channel(32);
    tokio::spawn(collection::run(Arc::clone(&config), tx));

    let mut plugged_in_since: Option<NaiveDateTime> = None;

    info!(
        training_count = model.training_count(),
        "rycharger started"
    );

    while let Some(event) = rx.recv().await {
        match event {
            Event::SessionComplete(session) => {
                db.insert_session(&session)?;
                model.update(&session, config.model.prediciton_horizon_mins);
                db.save_model(&ModelWeights::LogisticRegression(model.clone()))?;
                plugged_in_since = None;
                info!(
                    training_count = model.training_count(),
                    "model updated with new session"
                );
            }
            Event::Tick(snap) => {
                if snap.on_ac && plugged_in_since.is_none() {
                    plugged_in_since = Some(Local::now().naive_local());
                } else if !snap.on_ac {
                    plugged_in_since = None;
                }

                if snap.on_ac {
                    set_charge_target(&config, &model, plugged_in_since);
                }
            }
        }
    }

    Ok(())
}

/// Load saved model weights, or retrain from stored sessions if available.
fn load_or_train_model(db: &Database, config: &Config) -> Result<LogisticRegression> {
    if let Some(ModelWeights::LogisticRegression(model)) = db.load_model()? {
        info!(
            training_count = model.training_count(),
            "loaded model from disk"
        );
        return Ok(model);
    }

    let sessions = db.load_sessions()?;
    let mut model = LogisticRegression::new();

    if !sessions.is_empty() {
        info!(sessions = sessions.len(), "retraining model from history");
        model.train(&sessions, config.model.prediciton_horizon_mins);
        db.save_model(&ModelWeights::LogisticRegression(model.clone()))?;
    }

    Ok(model)
}

/// Decide whether to charge to full or hold, and write the threshold to sysfs.
fn set_charge_target(
    config: &Config,
    model: &LogisticRegression,
    plugged_in_since: Option<NaiveDateTime>,
) {
    let target = if model.training_count() < config.model.min_training_sessions {
        config.battery.hold_percent
    } else {
        let now = Local::now().naive_local();
        let minutes_plugged = plugged_in_since
            .map(|t| now.signed_duration_since(t).num_minutes() as f64)
            .unwrap_or(0.0);

        let features = Features::extract(now, minutes_plugged, model.avg_session_length_mins());
        let prediction = model.predict(&features, config.model.charge_threshold);

        if prediction.should_charge_to_full {
            info!(
                prob = format!("{:.2}", prediction.unplug_probability),
                "model predicts unplug soon — charging to full"
            );
            config.battery.full_percent
        } else {
            config.battery.hold_percent
        }
    };

    if let Err(e) = write_charge_threshold(&config.battery.device, target) {
        warn!("failed to set charge threshold: {e:#}");
    }
}

fn write_charge_threshold(device: &str, percent: u8) -> Result<()> {
    let path = format!("/sys/class/power_supply/{device}/charge_control_end_threshold");
    std::fs::write(&path, percent.to_string())
        .with_context(|| format!("writing charge threshold to {path}"))?;
    Ok(())
}
