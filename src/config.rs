use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Config {
    #[serde(default)]
    pub battery: BatteryConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub daemon: DaemonConfig,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct BatteryConfig {
    /// Battery device to use in /sys/class/power_supply/<device>.
    /// Empty string triggers auto-detection on startup.
    pub device: String,

    /// Percentage at which to hold charge
    pub hold_percent: u8,

    /// Percentage at which to consider the battery full
    pub full_percent: u8,
}

impl Default for BatteryConfig {
    fn default() -> Self {
        Self {
            device: String::new(),
            hold_percent: 80,
            full_percent: 100,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    LogisticRegression,
    #[default]
    GradientBoostedTree,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Which model implementation to use
    pub model_type: ModelType,

    /// Probability threshold above which we charge to full
    pub charge_threshold: f64,

    /// How far ahead (in minutes) to predict whether the user will unplug
    pub prediction_horizon_mins: u64,

    /// Minimum number of recorded sessions before the model starts making predictions
    pub min_training_sessions: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct DaemonConfig {
    /// How often (in seconds) to poll battery state and run predictions
    pub poll_interval_secs: u64,

    /// Path to SQLite database for session logging
    pub db_path: PathBuf,
}

/// Scans `/sys/class/power_supply/` for a battery device that exposes
/// `charge_control_end_threshold` and returns the first match (alphabetically).
fn detect_battery_device() -> anyhow::Result<String> {
    let mut candidates: Vec<String> = std::fs::read_dir("/sys/class/power_supply")?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .join("charge_control_end_threshold")
                .exists()
        })
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();

    candidates.sort();

    candidates.into_iter().next().ok_or_else(|| {
        anyhow::anyhow!(
            "Could not auto-detect a battery device: no entry under \
             /sys/class/power_supply/ exposes charge_control_end_threshold. \
             Set battery.device manually in the config file."
        )
    })
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::default(),
            charge_threshold: 0.7,
            prediction_horizon_mins: 60,
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

        let mut config: Config = if path.exists() {
            let contents = std::fs::read_to_string(&path)?;
            toml::from_str(&contents)?
        } else {
            Config::default()
        };

        if config.battery.device.is_empty() {
            let device = detect_battery_device()?;
            tracing::info!("Auto-detected battery device: {device}");
            config.battery.device = device;
            if let Err(e) = config.save() {
                tracing::warn!(
                    "Could not persist config to {}: {e}. Auto-detection will run again next startup.",
                    Self::config_file().display()
                );
            }
        }

        Ok(config)
    }

    fn save(&self) -> anyhow::Result<()> {
        let dir = Self::config_directory();
        std::fs::create_dir_all(&dir)?;

        let contents = toml::to_string_pretty(self)?;
        std::fs::write(Self::config_file(), contents)?;
        Ok(())
    }
}
