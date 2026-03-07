use std::sync::Arc;

use rycharger::config::Config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let config = Arc::new(Config::load()?);

    rycharger::daemon::run(config).await
}
