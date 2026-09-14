"""Calibradores NumPy: Platt, isotonic PAVA, ECE (sin sklearn)."""

from __future__ import annotations

import numpy as np
from baseline.calibration import (
    apply_isotonic_interpolation,
    apply_platt_scaling,
    apply_temperature_scaling,
    brier_score_binary,
    expected_calibration_error_binary,
    fit_isotonic_regression_on_probabilities,
    fit_platt_scaling_on_probabilities,
)


def test_platt_identity_matches_raw() -> None:
    p = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
    out = apply_platt_scaling(p, 1.0, 0.0)
    assert np.allclose(out, p, atol=1e-9)


def test_temperature_one_matches_raw() -> None:
    p = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
    out = apply_temperature_scaling(p, 1.0)
    assert np.allclose(out, p, atol=1e-9)


def test_platt_reduces_ece_on_biased_scores() -> None:
    y = np.concatenate([np.zeros(240, dtype=np.int32), np.ones(80, dtype=np.int32)])
    p = np.concatenate([np.full(240, 0.48), np.full(80, 0.62)])
    ece_raw = expected_calibration_error_binary(y, p, n_bins=10)
    a, b, _diag = fit_platt_scaling_on_probabilities(y, p)
    q = apply_platt_scaling(p, a, b)
    ece_platt = expected_calibration_error_binary(y, q, n_bins=10)
    assert ece_platt < ece_raw
    assert brier_score_binary(y, q) <= brier_score_binary(y, p) + 1e-12


def test_isotonic_pava_is_nondecreasing() -> None:
    rng = np.random.default_rng(42)
    p = rng.uniform(0.05, 0.95, size=80)
    y = (p + rng.normal(0, 0.2, size=80) > 0.5).astype(np.float64)
    x_knots, y_knots, diag = fit_isotonic_regression_on_probabilities(y, p)
    assert int(diag["n_knots"]) >= 1
    assert np.all(np.diff(y_knots) >= -1e-12)
    fitted = apply_isotonic_interpolation(np.sort(p), x_knots, y_knots)
    assert np.all(np.diff(fitted) >= -1e-12)
