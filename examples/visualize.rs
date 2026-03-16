use std::path::PathBuf;

use anyhow::{bail, Result};
use chrono::NaiveDate;
use rycharger::{model::Features, storage::Database};

const DAYS: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const BLOCKS: [char; 9] = [' ', '░', '░', '▒', '▒', '▓', '▓', '█', '█'];

const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";

/// Map probability (0.0–1.0) to a colored pair of block chars on an absolute scale.
fn heatmap_cell(prob: f64, threshold: f64) -> String {
    let idx = (prob * 8.0).round() as usize;
    let blk = BLOCKS[idx.min(8)];
    let clr = color_for(prob, threshold);
    format!("{clr}{blk}{blk}{RESET}")
}

fn color_for(prob: f64, threshold: f64) -> &'static str {
    if prob >= threshold {
        "\x1b[1;91m" // bold bright red — would charge to full
    } else if prob >= threshold * 0.6 {
        "\x1b[33m" // yellow — moderate
    } else if prob >= 0.05 {
        "\x1b[36m" // cyan — low
    } else {
        "\x1b[2;37m" // dim white — near zero
    }
}

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

    let model_weights = db.load_model()?;
    let Some(weights) = model_weights else {
        bail!(
            "No trained model found in {}. \
             Run rycharger for a while to collect sessions first.",
            db_path.display()
        );
    };

    let model = weights.as_model();
    let threshold = config.model.charge_threshold;
    let avg_session = model.avg_session_length_mins();

    // Use a Monday as reference so weekday index lines up
    let ref_monday = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(); // a Monday

    // Pre-compute all probabilities for the grid
    let mut grid = [[0.0_f64; 24]; 7]; // [day][hour]
    let mut global_min = f64::MAX;
    let mut global_max = f64::MIN;

    for day_idx in 0..7 {
        let date = ref_monday + chrono::Duration::days(day_idx as i64);
        for hour in 0..24 {
            let mut sum = 0.0;
            for half in [0, 30] {
                let dt = date.and_hms_opt(hour as u32, half, 0).unwrap();
                let features = Features::extract(dt, 0.0, avg_session);
                let pred = model.predict(&features, threshold);
                sum += pred.unplug_probability;
            }
            let prob = sum / 2.0;
            grid[day_idx][hour] = prob;
            global_min = global_min.min(prob);
            global_max = global_max.max(prob);
        }
    }

    // Print header
    println!();
    println!("{BOLD}  rycharger — unplug probability heatmap{RESET}");
    println!(
        "{DIM}  Threshold: {:.0}%  |  Horizon: {} min  |  Sessions: {}  |  Avg session: {:.0} min{RESET}",
        threshold * 100.0,
        config.model.prediction_horizon_mins,
        model.training_count(),
        avg_session
    );
    println!(
        "{DIM}  Probability range: {:.1}% — {:.1}%{RESET}",
        global_min * 100.0,
        global_max * 100.0,
    );
    println!();

    // Hour header
    print!("      ");
    for h in 0..24 {
        if h % 3 == 0 {
            print!("{DIM}{h:>2}{RESET}");
        } else {
            print!("  ");
        }
    }
    println!();

    // Heatmap rows
    for day_idx in 0..7 {
        print!("  {BOLD}{}{RESET}  ", DAYS[day_idx]);

        for hour in 0..24 {
            print!("{}", heatmap_cell(grid[day_idx][hour], threshold));
        }

        // Show peak hour for this day
        let (peak_hour, &peak_prob) = grid[day_idx]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        println!(
            "  {DIM}peak {peak_hour:>2}:00 ({:.1}%){RESET}",
            peak_prob * 100.0
        );
    }

    println!();

    // Legend
    print!("  {DIM}0%{RESET} ");
    for i in 0..=8 {
        let p = i as f64 / 8.0;
        let clr = color_for(p, threshold);
        print!("{clr}{}{RESET}", BLOCKS[i]);
    }
    print!(" {DIM}100%{RESET}");
    println!(
        "    {}\x1b[1;91m|{RESET}{} = charge to full (>= {:.0}%){RESET}",
        DIM,
        DIM,
        threshold * 100.0
    );
    println!();

    // Top-5 highest prediction slots
    let mut slots: Vec<(usize, usize, f64)> = Vec::new();
    for day_idx in 0..7 {
        for hour in 0..24 {
            slots.push((day_idx, hour, grid[day_idx][hour]));
        }
    }
    slots.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    println!("  {BOLD}Top predicted unplug times:{RESET}");
    for &(day, hour, prob) in slots.iter().take(5) {
        let marker = if prob >= threshold {
            "\x1b[1;91m*\x1b[0m"
        } else {
            " "
        };
        println!(
            "   {marker} {} {:>2}:00  {:.1}%",
            DAYS[day],
            hour,
            prob * 100.0
        );
    }
    println!();

    Ok(())
}
