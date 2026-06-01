"""Test-time augmentation ligera (solo API)."""

from __future__ import annotations

import numpy as np


def horizontal_flip_rgb(rgb: np.ndarray) -> np.ndarray:
    return np.flip(rgb, axis=1).copy()


def average_raw_probability_with_optional_tta(
    predictor,
    rgb_uint8: np.ndarray,
    *,
    tta_enabled: bool,
) -> float:
    p0 = float(predictor.predict_from_rgb(rgb_uint8))
    if not tta_enabled:
        return p0
    p1 = float(predictor.predict_from_rgb(horizontal_flip_rgb(rgb_uint8)))
    return (p0 + p1) / 2.0
