use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Config {
    pub battery: BatteryConfig,
    pub model: ModelConfig,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    /// Percentage at which to hold charge
    pub hold_charge: u8,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct BatteryConfig {
    /// Battery device to use in /sys/class/power_supply/<device>
    pub device: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self { hold_charge: 80 }
    }
}

impl Default for BatteryConfig {
    fn default() -> Self {
        Self {
            device: "BAT0".to_string(),
        }
    }
}

impl Config {
    fn config_directory() -> PathBuf {
        dirs::config_dir()
            .expect("Could not find config directory")
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
