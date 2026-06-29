#!/usr/bin/env bash
# MediaPipe Hand Landmarker en Linux headless (CI ubuntu + referencia local).
set -euo pipefail
if [ "$(uname -s)" != Linux ]; then
  echo "install-mediapipe-system-libs: skip (not Linux)"
  exit 0
fi
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  libgl1-mesa-glx \
  libgles2-mesa \
  libegl1-mesa
