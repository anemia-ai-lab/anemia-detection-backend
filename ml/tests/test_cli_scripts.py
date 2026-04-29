"""Validación ligera de CLIs G8/G10 (``--help`` sin ejecutar inferencia)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = REPO_ROOT / "ml"


@pytest.mark.parametrize(
    ("script", "needle"),
    [
        ("scripts/run_tflite_inference.py", "tflite"),
        ("scripts/generate_gradcam.py", "mobilenet"),
    ],
)
def test_cli_help_exits_zero(script: str, needle: str) -> None:
    """Imports de TensorFlow están dentro de ``main()``; ``--help`` no debe cargar el modelo."""
    path = ML_ROOT / script
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    r = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=str(ML_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    combined = (r.stdout + r.stderr).lower()
    assert needle in combined
