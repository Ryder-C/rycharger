use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{Local, NaiveDateTime};
use tokio::sync::mpsc;
use tokio::time::{self, Duration};
use tracing::{debug, info, warn};

use crate::config::Config;
use crate::model::Session;

/// Battery status as reported by sysfs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatteryStatus {
    Charging,
    Discharging,
    NotCharging,
    Full,
    Unknown,
}

impl BatteryStatus {
    fn parse(s: &str) -> Self {
        match s.trim() {
            "Charging" => Self::Charging,
            "Discharging" => Self::Discharging,
            "Not charging" => Self::NotCharging,
            "Full" => Self::Full,
            _ => Self::Unknown,
        }
    }

    /// Whether AC power is connected
    pub fn on_ac(self) -> bool {
        matches!(self, Self::Charging | Self::NotCharging | Self::Full)
    }
}

/// A snapshot of the battery at a point in time
#[derive(Debug, Clone)]
pub struct BatterySnapshot {
    pub status: BatteryStatus,
    pub capacity: u8,
    pub on_ac: bool,
}

/// Events emitted by the collector
pub enum Event {
    /// A periodic battery reading (every poll tick)
    Tick(BatterySnapshot),
    /// A completed charge session (charger was unplugged)
    SessionComplete(Session),
}

fn sysfs_path(device: &str) -> PathBuf {
    PathBuf::from(format!("/sys/class/power_supply/{device}"))
}

async fn read_battery(device: &str) -> Result<BatterySnapshot> {
    let base = sysfs_path(device);

    let status_raw = tokio::fs::read_to_string(base.join("status"))
        .await
        .context("reading battery status")?;

    let capacity_raw = tokio::fs::read_to_string(base.join("capacity"))
        .await
        .context("reading battery capacity")?;

    let capacity: u8 = capacity_raw
        .trim()
        .parse()
        .context("parsing battery capacity")?;

    let status = BatteryStatus::parse(&status_raw);
    Ok(BatterySnapshot {
        on_ac: status.on_ac(),
        status,
        capacity,
    })
}

/// Polls battery state and emits [`Event`]s: ticks every interval, sessions on unplug.
pub async fn run(config: Arc<Config>, tx: mpsc::Sender<Event>) -> Result<()> {
    let mut interval = time::interval(Duration::from_secs(config.daemon.poll_interval_secs));
    let mut was_on_ac = false;
    let mut session_start: Option<NaiveDateTime> = None;

    // Read initial state so we don't falsely detect a "plug-in" on startup
    match read_battery(&config.battery.device).await {
        Ok(snap) => {
            was_on_ac = snap.on_ac;
            if was_on_ac {
                info!(capacity = snap.capacity, "started while on AC power");
                session_start = Some(Local::now().naive_local());
            } else {
                info!(capacity = snap.capacity, "started on battery power");
            }
        }
        Err(e) => warn!("initial battery read failed: {e:#}"),
    }

    loop {
        interval.tick().await;

        let snap = match read_battery(&config.battery.device).await {
            Ok(s) => s,
            Err(e) => {
                warn!("battery read failed, will retry: {e:#}");
                continue;
            }
        };

        debug!(status = ?snap.status, capacity = snap.capacity, snap.on_ac);

        // Detect transitions
        match (was_on_ac, snap.on_ac) {
            (false, true) => {
                info!(capacity = snap.capacity, "charger plugged in");
                session_start = Some(Local::now().naive_local());
            }
            (true, false) => {
                if let Some(plugged_in_at) = session_start.take() {
                    let session = Session {
                        plugged_in_at,
                        unplugged_at: Local::now().naive_local(),
                    };
                    info!(
                        duration_mins = session
                            .unplugged_at
                            .signed_duration_since(session.plugged_in_at)
                            .num_minutes(),
                        "charger unplugged, session recorded"
                    );
                    send(&tx, Event::SessionComplete(session)).await?;
                }
            }
            _ => {}
        }

        was_on_ac = snap.on_ac;

        // Always emit a tick so the main loop can run charge control
        send(&tx, Event::Tick(snap)).await?;
    }
}

async fn send(tx: &mpsc::Sender<Event>, event: Event) -> Result<()> {
    if tx.send(event).await.is_err() {
        info!("event receiver dropped, stopping collector");
        anyhow::bail!("receiver dropped");
    }
    Ok(())
}
