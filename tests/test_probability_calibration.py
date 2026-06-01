"""Calibración numérica en inferencia (sin TensorFlow)."""

from backend.core.config import (
    INFERENCE_CALIBRATION_OPERATIONAL_THRESHOLD_DEFAULT,
    INFERENCE_CALIBRATION_TEMPERATURE_DEFAULT,
)
from backend.inference.probability_calibration import (
    apply_temperature_calibration,
    binary_prediction_from_threshold,
)


def test_temperature_one_matches_raw_probability() -> None:
    for p in (0.01, 0.2, 0.5, 0.8, 0.99):
        calibrated = apply_temperature_calibration(p, 1.0)
        assert abs(calibrated - p) < 1e-9


def test_binary_prediction_inclusive_threshold() -> None:
    assert binary_prediction_from_threshold(0.2, 0.168) == 1
    assert binary_prediction_from_threshold(0.168, 0.168) == 1
    assert binary_prediction_from_threshold(0.16799, 0.168) == 0


def test_default_constants_are_v2_ensemble_values() -> None:
    assert INFERENCE_CALIBRATION_TEMPERATURE_DEFAULT == 1.405026093389256
    assert INFERENCE_CALIBRATION_OPERATIONAL_THRESHOLD_DEFAULT == 0.3815443834698594


def test_midpoint_stays_half_with_thesis_temperature() -> None:
    p = apply_temperature_calibration(0.5, INFERENCE_CALIBRATION_TEMPERATURE_DEFAULT)
    assert abs(p - 0.5) < 1e-6
