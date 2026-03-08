use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Config {
    pub battery: BatteryConfig,
    pub model: ModelConfig,
    pub daemon: DaemonConfig,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct BatteryConfig {
    /// Battery device to use in /sys/class/power_supply/<device>
    pub device: String,

    /// Percentage at which to hold charge
    pub hold_percent: u8,

    /// Percentage at which to consider the battery full
    pub full_percent: u8,
}

#[derive(Debug, Default, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    LogisticRegression,
    #[default]
    GradientBoostedTree,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    /// Which model implementation to use
    pub model_type: ModelType,

    /// Probability threshold above which we charge to full
    pub charge_threshold: f64,

    /// How far ahead (in minutes) to predict whether the user will unplug
    pub prediciton_horizon_mins: u64,

    /// Minimum number of recorded sessions before the model starts making predicitons
    pub min_training_sessions: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DaemonConfig {
    /// How often (in seconds) to poll battery state and run predictions
    pub poll_interval_secs: u64,

    /// Path to SQLite database for session logging
    pub db_path: PathBuf,
}

impl Default for BatteryConfig {
    fn default() -> Self {
        Self {
            device: "BAT0".to_string(),
            hold_percent: 80,
            full_percent: 100,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::default(),
            charge_threshold: 0.7,
            prediciton_horizon_mins: 60,
            min_training_sessions: 14,
        }
    }
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            poll_interval_secs: 60,
            db_path: Config::data_directory().join("rycharger.db"),
        }
    }
}

impl Config {
    fn config_directory() -> PathBuf {
        dirs::config_dir()
            .expect("Could not find config directory")
            .join("rycharger")
    }

    fn data_directory() -> PathBuf {
        dirs::data_dir()
            .expect("Could not find data directory")
            .join("rycharger")
    }

    fn config_file() -> PathBuf {
        Self::config_directory().join("config.toml")
    }

    pub fn load() -> anyhow::Result<Self> {
        let path = Self::config_file();

        if path.exists() {
            let contents = std::fs::read_to_string(&path)?;
            let config = toml::from_str(&contents)?;
            Ok(config)
        } else {
            let config = Self::default();
            config.save()?;
            Ok(config)
        }
    }

    fn save(&self) -> anyhow::Result<()> {
        let dir = Self::config_directory();
        std::fs::create_dir_all(&dir)?;

        let contents = toml::to_string_pretty(self)?;
        std::fs::write(Self::config_file(), contents)?;
        Ok(())
    }
}
