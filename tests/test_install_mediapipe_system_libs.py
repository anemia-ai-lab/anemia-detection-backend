"""Tests for install_mediapipe_system_libs.sh package selection logic."""

from __future__ import annotations

import subprocess
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install_mediapipe_system_libs.sh"


def _picker_functions() -> str:
    text = _SCRIPT.read_text(encoding="utf-8")
    start = text.index("apt_pkg_installable()")
    end = text.index('echo "install-mediapipe-system-libs: ${PACKAGES[*]}"')
    return text[start:end]


def _run_picker(*apt_policy_lines: str) -> str:
    mock = "\n".join(apt_policy_lines)
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"""
{mock}
{_picker_functions()}
pick_packages
echo "${{PACKAGES[*]}}"
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_ubuntu_noble_picks_modern_gl_packages() -> None:
    """libgl1-mesa-glx aparece en apt pero no es instalable (Ubuntu 24.04)."""
    out = _run_picker(
        "apt-cache() {",
        '  if [ "$1" = policy ]; then',
        '    case "$2" in',
        '      libgl1-mesa-glx|libgles2-mesa|libegl1-mesa) echo "Candidate: (none)" ;;',
        '      libgl1|libglx-mesa0|libgles2|libegl1) echo "Candidate: 24.0.5-1ubuntu1" ;;',
        '      *) echo "Candidate: (none)" ;;',
        "    esac",
        "  fi",
        "}",
    )
    assert out == "libgl1 libglx-mesa0 libgles2 libegl1"


def test_debian_bookworm_picks_legacy_gl_packages() -> None:
    out = _run_picker(
        "apt-cache() {",
        '  if [ "$1" = policy ]; then',
        '    case "$2" in',
        '      libgl1-mesa-glx|libgles2-mesa|libegl1-mesa) echo "Candidate: 22.3.6-1+deb12u1" ;;',
        '      *) echo "Candidate: (none)" ;;',
        "    esac",
        "  fi",
        "}",
    )
    assert out == "libgl1-mesa-glx libgles2-mesa libegl1-mesa"
