#!/usr/bin/env bash
# MediaPipe Hand Landmarker en Linux headless.
# Debian bookworm (Docker prod) y Ubuntu 24.04+ (GitHub Actions) usan nombres distintos.
set -euo pipefail

if [ "$(uname -s)" != Linux ]; then
  echo "install-mediapipe-system-libs: skip (not Linux)"
  exit 0
fi

apt_pkg_installable() {
  local candidate
  candidate="$(apt-cache policy "$1" 2>/dev/null | awk '/Candidate:/ {print $2; exit}')"
  [ -n "${candidate:-}" ] && [ "${candidate}" != "(none)" ]
}

pick_packages() {
  PACKAGES=()
  # apt-cache show puede listar libgl1-mesa-glx en Ubuntu aunque no sea instalable.
  if apt_pkg_installable libgl1-mesa-glx && apt_pkg_installable libgles2-mesa; then
    PACKAGES+=(libgl1-mesa-glx libgles2-mesa libegl1-mesa)
  else
    PACKAGES+=(libgl1 libglx-mesa0 libgles2 libegl1)
  fi
}

pick_packages
echo "install-mediapipe-system-libs: ${PACKAGES[*]}"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${PACKAGES[@]}"
