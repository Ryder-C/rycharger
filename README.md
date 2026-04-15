# rycharger

A smart battery charge management daemon for Linux laptops. rycharger learns your usage patterns and automatically adjusts your charge threshold.

> [!NOTE]
> rycharger won't make any charging decisions until it has sufficient training sessions. See [Configuration](#configuration)

## How it works

rycharger polls your battery every minute and records charging sessions. Over time, it trains a machine learning model on your plug/unplug history. When you're plugged in, it predicts the probability you'll unplug within the next hour and sets your charge threshold accordingly:

- **Likely to stay plugged in** &rarr; holds charge at ~80%, reducing battery wear
- **Likely to unplug soon** &rarr; charges to 100% so you leave with a full battery

Predictions use features like time of day, day of week, and how long you've already been plugged in. The model improves as it collects more sessions.

## Requirements

- Linux with a battery that exposes `charge_control_end_threshold` via sysfs (most ThinkPads, MacBooks (asahi), some other laptops)

Check if your battery supports threshold control:

```bash
ls /sys/class/power_supply/BAT0/charge_control_end_threshold
```

## Installation

### Linux

```bash
curl -fsSL https://raw.githubusercontent.com/Ryder-C/rycharger/main/install.sh | sh
```

This installs the binary to `~/.local/bin/rycharger`, sets up a udev rule (requires sudo) so your user can write to the sysfs charge threshold without root, and enables a systemd user service that starts automatically on login.

### Docker

```bash
docker run -d \
  --name rycharger \
  --restart unless-stopped \
  --privileged \
  -v /sys/class/power_supply:/sys/class/power_supply \
  -v rycharger-data:/var/lib/rycharger \
  ghcr.io/ryder-c/rycharger:latest
```

The image stores both the config file and the SQLite database under `/var/lib/rycharger/`, so the single named volume persists everything.

> [!WARNING]
> `--privileged` is required to write to sysfs. If you prefer not to use it, you can grant write access via the udev rule in the Linux install script and mount only the specific battery path instead.

Or with Docker Compose:

```yaml
services:
  rycharger:
    image: ghcr.io/ryder-c/rycharger:latest
    restart: unless-stopped
    privileged: true
    volumes:
      - /sys/class/power_supply:/sys/class/power_supply
      - rycharger-data:/var/lib/rycharger

volumes:
  rycharger-data:
```

```bash
docker compose up -d
```

### NixOS

Add rycharger to your flake inputs:

```nix
inputs.rycharger.url = "github:Ryder-C/rycharger";
```

Then enable the NixOS module:

```nix
{ inputs, ... }:
{
  imports = [ inputs.rycharger.nixosModules.default ];

  services.rycharger = {
    enable = true;
    # settings.battery.device = "BAT0"; # Optional override; auto-detected by default
  };
}
```

The service runs as a systemd unit with the permissions it needs to write to sysfs. Config and data are stored under `/var/lib/rycharger/`.

## Configuration

On first run, rycharger scans `/sys/class/power_supply/` for a device that exposes `charge_control_end_threshold`, picks the first match alphabetically, and writes `~/.config/rycharger/config.toml` with sensible defaults:

```toml
[battery]
device = "BAT0"       # Auto-detected on first run; edit if the wrong battery was picked
hold_percent = 80     # Charge limit when staying plugged in
full_percent = 100    # Charge limit when about to unplug

[model]
model_type = "gradient_boosted_tree"  # "gradient_boosted_tree" or "logistic_regression"
charge_threshold = 0.7                # Probability needed to trigger a full charge
prediction_horizon_mins = 60          # How far ahead to predict unplugging
min_training_sessions = 14            # Sessions required before predictions kick in

[daemon]
poll_interval_secs = 60
db_path = "~/.local/share/rycharger/rycharger.db"
```

## Data storage

Paths follow the [XDG base-dir spec](https://specifications.freedesktop.org/basedir-spec/latest/), so they differ by install method:

| Install      | Config                            | Database                                |
| ------------ | --------------------------------- | --------------------------------------- |
| Linux script | `~/.config/rycharger/config.toml` | `~/.local/share/rycharger/rycharger.db` |
| NixOS module | `/etc/rycharger/config.toml`      | `/var/lib/rycharger/rycharger.db`       |
| Docker       | `/var/lib/rycharger/config.toml`  | `/var/lib/rycharger/rycharger.db`       |

## License

MIT
