#!/usr/bin/env bash
set -euo pipefail

REPO="Ryder-C/rycharger"
BIN_DIR="${HOME}/.local/bin"
SERVICE_DIR="${HOME}/.config/systemd/user"
UDEV_RULE="/etc/udev/rules.d/99-rycharger.rules"

# Detect architecture
ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64)  ASSET="rycharger-x86_64-linux" ;;
  aarch64) ASSET="rycharger-aarch64-linux" ;;
  *)
    echo "Unsupported architecture: ${ARCH}"
    exit 1
    ;;
esac

echo "Fetching latest release..."
URL="https://github.com/${REPO}/releases/latest/download/${ASSET}"

mkdir -p "${BIN_DIR}"
curl -fsSL "${URL}" -o "${BIN_DIR}/rycharger"
chmod +x "${BIN_DIR}/rycharger"
echo "Installed to ${BIN_DIR}/rycharger"

# udev rule for sysfs write access
if [ ! -f "${UDEV_RULE}" ]; then
  echo "Installing udev rule (requires sudo)..."
  echo 'SUBSYSTEM=="power_supply", ATTR{type}=="Battery", RUN+="/bin/chmod a+w /sys/class/power_supply/%k/charge_control_end_threshold"' \
    | sudo tee "${UDEV_RULE}" > /dev/null
  sudo udevadm control --reload
  sudo udevadm trigger
  echo "udev rule installed"
else
  echo "udev rule already exists, skipping"
fi

# systemd user service
mkdir -p "${SERVICE_DIR}"
cat > "${SERVICE_DIR}/rycharger.service" <<EOF
[Unit]
Description=rycharger battery daemon

[Service]
ExecStart=${BIN_DIR}/rycharger
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now rycharger
echo "rycharger service enabled and started"
