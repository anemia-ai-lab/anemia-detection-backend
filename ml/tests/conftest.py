"""Pytest para el subproyecto ML (TensorFlow + artefactos locales)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = REPO_ROOT / "ml"
for _p in (str(REPO_ROOT), str(ML_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("tensorflow")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

# Semilla global explícita para suites ML (determinismo sin depender del orden de tests).
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

MODEL_KERAS = ML_ROOT / "artifacts" / "models" / "baseline_mobilenetv2.keras"
MODEL_TFLITE = ML_ROOT / "artifacts" / "models" / "baseline_mobilenetv2_v1.tflite"
TFLITE_METADATA = ML_ROOT / "artifacts" / "models" / "baseline_mobilenetv2_v1.metadata.json"


@pytest.fixture
def keras_model_path() -> Path:
    if not MODEL_KERAS.is_file():
        pytest.skip(f"Artefacto Keras no encontrado: {MODEL_KERAS}")
    return MODEL_KERAS


@pytest.fixture
def tflite_paths() -> tuple[Path, Path]:
    if not MODEL_TFLITE.is_file() or not TFLITE_METADATA.is_file():
        pytest.skip("Artefactos TFLite o metadatos no encontrados")
    return MODEL_TFLITE, TFLITE_METADATA
