"""Artefactos oficiales v2 Ghana versionados en git (whitelist .gitignore)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.tests.conftest import ML_ROOT

OFFICIAL_KERAS = tuple(
    ML_ROOT / "artifacts" / "models" / f"baseline_mobilenetv2_ghana_augmented_seed{seed}.keras"
    for seed in (42, 123, 456)
)
OFFICIAL_TFLITE = tuple(
    ML_ROOT / "artifacts" / "models" / f"baseline_mobilenetv2_ghana_augmented_seed{seed}.tflite"
    for seed in (42, 123, 456)
)
OFFICIAL_TFLITE_METADATA = tuple(
    ML_ROOT
    / "artifacts"
    / "models"
    / f"baseline_mobilenetv2_ghana_augmented_seed{seed}.metadata.json"
    for seed in (42, 123, 456)
)
ENSEMBLE_METADATA = (
    ML_ROOT / "artifacts" / "models" / "baseline_mobilenetv2_ghana_ensemble.metadata.json"
)
CALIBRATION_JSON = ML_ROOT / "artifacts" / "runs" / "calibration_ensemble_ghana_v2.json"
CALIBRATION_MD = ML_ROOT / "artifacts" / "runs" / "calibration_ensemble_ghana_v2.md"


@pytest.mark.parametrize("path", OFFICIAL_KERAS)
def test_official_keras_present(path: Path) -> None:
    assert path.is_file(), f"Falta artefacto Keras oficial: {path.name}"


@pytest.mark.parametrize("path", OFFICIAL_TFLITE)
def test_official_tflite_present(path: Path) -> None:
    assert path.is_file(), f"Falta artefacto TFLite oficial: {path.name}"


@pytest.mark.parametrize("path", OFFICIAL_TFLITE_METADATA)
def test_official_tflite_metadata_present(path: Path) -> None:
    assert path.is_file(), f"Falta metadata TFLite oficial: {path.name}"


def test_official_ensemble_metadata_present() -> None:
    assert ENSEMBLE_METADATA.is_file(), "Falta baseline_mobilenetv2_ghana_ensemble.metadata.json"


def test_official_calibration_run_present() -> None:
    assert CALIBRATION_JSON.is_file(), "Falta calibration_ensemble_ghana_v2.json"
    assert CALIBRATION_MD.is_file(), "Falta calibration_ensemble_ghana_v2.md"
