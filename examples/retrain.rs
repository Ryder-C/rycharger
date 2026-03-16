use std::path::PathBuf;

use anyhow::{bail, Result};
use rycharger::model::ChargeModel as _;
use rycharger::storage::Database;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let config_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/etc/rycharger/config.toml"));
    let db_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/var/lib/rycharger/rycharger.db"));

    let config: rycharger::config::Config = if config_path.exists() {
        toml::from_str(&std::fs::read_to_string(&config_path)?)?
    } else {
        bail!("Config not found at {}", config_path.display());
    };

    let db = Database::open(&db_path)?;
    let sessions = db.load_sessions()?;

    if sessions.is_empty() {
        bail!("No sessions found in {}", db_path.display());
    }

    println!("Loaded {} sessions from {}", sessions.len(), db_path.display());

    let mut model = rycharger::model::GradientBoostedTree::new();
    model.train(&sessions, config.model.prediction_horizon_mins);

    let weights = rycharger::storage::ModelWeights::GradientBoostedTree(model);
    db.save_model(&weights)?;

    println!("Model retrained and saved.");

    Ok(())
}
