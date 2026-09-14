"""Calibración numérica en inferencia (sin TensorFlow)."""

from backend.core.config import (
    INFERENCE_CALIBRATION_OPERATIONAL_THRESHOLD_DEFAULT,
    INFERENCE_CALIBRATION_TEMPERATURE_DEFAULT,
)
from backend.inference.probability_calibration import (
    apply_platt_calibration,
    apply_probability_calibration,
    apply_temperature_calibration,
    binary_prediction_from_threshold,
    calibration_is_enabled,
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
    assert INFERENCE_CALIBRATION_TEMPERATURE_DEFAULT == 0.9443417710165931
    assert INFERENCE_CALIBRATION_OPERATIONAL_THRESHOLD_DEFAULT == 0.5780355600619943


def test_midpoint_stays_half_with_thesis_temperature() -> None:
    p = apply_temperature_calibration(0.5, INFERENCE_CALIBRATION_TEMPERATURE_DEFAULT)
    assert abs(p - 0.5) < 1e-6


def test_platt_identity_matches_raw() -> None:
    for p in (0.01, 0.2, 0.5, 0.8, 0.99):
        calibrated = apply_platt_calibration(p, 1.0, 0.0)
        assert abs(calibrated - p) < 1e-9
        via_dispatch = apply_probability_calibration(
            p, method="platt", platt_a=1.0, platt_b=0.0
        )
        assert abs(via_dispatch - p) < 1e-9


def test_calibration_is_enabled_flags() -> None:
    assert calibration_is_enabled(
        method="temperature", temperature=1.405, platt_a=1.0, platt_b=0.0
    )
    assert not calibration_is_enabled(
        method="temperature", temperature=1.0, platt_a=1.0, platt_b=0.0
    )
    assert calibration_is_enabled(
        method="platt", temperature=1.0, platt_a=0.8, platt_b=0.0
    )
