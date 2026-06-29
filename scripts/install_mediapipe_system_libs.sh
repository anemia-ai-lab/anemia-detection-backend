#!/usr/bin/env bash
# MediaPipe Hand Landmarker en Linux headless.
# Debian bookworm (Docker prod) y Ubuntu 24.04+ (GitHub Actions) usan nombres distintos.
set -euo pipefail

if [ "$(uname -s)" != Linux ]; then
  echo "install-mediapipe-system-libs: skip (not Linux)"
  exit 0
fi

pick_packages() {
  PACKAGES=()
  if apt-cache show libgl1-mesa-glx >/dev/null 2>&1; then
    PACKAGES+=(libgl1-mesa-glx libgles2-mesa libegl1-mesa)
  else
    # Ubuntu 23.10+ / noble: libgl1-mesa-glx obsoleto
    PACKAGES+=(libgl1 libglx-mesa0 libgles2 libegl1)
  fi
}

pick_packages
echo "install-mediapipe-system-libs: ${PACKAGES[*]}"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${PACKAGES[@]}"
