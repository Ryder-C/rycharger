use std::path::Path;

use anyhow::{Context, Result};
use chrono::NaiveDateTime;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::model::{ChargeModel, GradientBoostedTree, LogisticRegression, Session};

const DATETIME_FMT: &str = "%Y-%m-%d %H:%M:%S";

/// Tagged enum so we can support different model types in the future
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ModelWeights {
    LogisticRegression(LogisticRegression),
    GradientBoostedTree(GradientBoostedTree),
}

impl ModelWeights {
    pub fn as_model(&self) -> &dyn ChargeModel {
        match self {
            ModelWeights::LogisticRegression(m) => m,
            ModelWeights::GradientBoostedTree(m) => m,
        }
    }

    pub fn as_model_mut(&mut self) -> &mut dyn ChargeModel {
        match self {
            ModelWeights::LogisticRegression(m) => m,
            ModelWeights::GradientBoostedTree(m) => m,
        }
    }
}

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path).context("opening database")?;
        let db = Self { conn };
        db.migrate()?;
        Ok(db)
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS sessions (
                    id            INTEGER PRIMARY KEY,
                    plugged_in_at TEXT NOT NULL,
                    unplugged_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_weights (
                    id      INTEGER PRIMARY KEY CHECK (id = 1),
                    weights TEXT NOT NULL
                );",
            )
            .context("running migrations")?;
        Ok(())
    }

    pub fn insert_session(&self, session: &Session) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO sessions (plugged_in_at, unplugged_at) VALUES (?1, ?2)",
                (
                    session.plugged_in_at.format(DATETIME_FMT).to_string(),
                    session.unplugged_at.format(DATETIME_FMT).to_string(),
                ),
            )
            .context("inserting session")?;
        Ok(())
    }

    pub fn load_sessions(&self) -> Result<Vec<Session>> {
        let mut stmt = self
            .conn
            .prepare("SELECT plugged_in_at, unplugged_at FROM sessions ORDER BY id")?;

        let sessions = stmt
            .query_map([], |row| {
                let plugged_in: String = row.get(0)?;
                let unplugged: String = row.get(1)?;
                Ok((plugged_in, unplugged))
            })?
            .map(|row| {
                let (plugged_in, unplugged) = row.context("reading session row")?;
                Ok(Session {
                    plugged_in_at: NaiveDateTime::parse_from_str(&plugged_in, DATETIME_FMT)
                        .context("parsing plugged_in_at")?,
                    unplugged_at: NaiveDateTime::parse_from_str(&unplugged, DATETIME_FMT)
                        .context("parsing unplugged_at")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(sessions)
    }

    pub fn save_model(&self, weights: &ModelWeights) -> Result<()> {
        let json = serde_json::to_string(weights).context("serializing model weights")?;
        self.conn
            .execute(
                "INSERT INTO model_weights (id, weights) VALUES (1, ?1)
                 ON CONFLICT(id) DO UPDATE SET weights = ?1",
                [&json],
            )
            .context("saving model weights")?;
        Ok(())
    }

    pub fn load_model(&self) -> Result<Option<ModelWeights>> {
        let mut stmt = self
            .conn
            .prepare("SELECT weights FROM model_weights WHERE id = 1")?;

        let mut rows = stmt.query_map([], |row| {
            let json: String = row.get(0)?;
            Ok(json)
        })?;

        match rows.next() {
            Some(row) => {
                let json = row.context("reading model weights")?;
                let weights =
                    serde_json::from_str(&json).context("deserializing model weights")?;
                Ok(Some(weights))
            }
            None => Ok(None),
        }
    }
}
