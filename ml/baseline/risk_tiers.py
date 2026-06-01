"""Umbrales de riesgo en tres niveles (bajo / medio / alto) sobre probabilidad calibrada."""

from __future__ import annotations

import numpy as np

from baseline.evaluation import youden_optimal_threshold


def risk_tier_thresholds_from_validation(
    y_val: np.ndarray,
    p_cal_val: np.ndarray,
    *,
    low_percentile_on_negatives: float = 90.0,
) -> dict[str, float | str]:
    """
    - ``high_lower``: τ alto (Youden en val calibrado) — por encima = riesgo alto.
    - ``low_upper``: percentil de ``p_cal`` en negativos de val — por debajo = riesgo bajo.
    """
    y_val = np.asarray(y_val).astype(int, copy=False)
    p_cal_val = np.asarray(p_cal_val, dtype=np.float64, copy=False)
    high_lower, youden_j = youden_optimal_threshold(y_val, p_cal_val)

    neg_mask = y_val == 0
    if int(np.sum(neg_mask)) == 0:
        low_upper = float(high_lower) * 0.5
        low_source = "fallback_half_high_lower_no_negatives_in_val"
    else:
        low_upper = float(np.percentile(p_cal_val[neg_mask], low_percentile_on_negatives))
        low_source = f"p{low_percentile_on_negatives:.0f}_negatives_calibrated_validation"

    if low_upper >= high_lower:
        low_upper = float(high_lower) * 0.85
        low_source = f"{low_source}_clamped_below_high_lower"

    return {
        "low_upper": float(low_upper),
        "high_lower": float(high_lower),
        "youden_j_on_validation": float(youden_j),
        "low_upper_source": low_source,
        "high_lower_source": "roc_youden_max_j_on_validation_calibrated",
    }


def risk_tier_from_probability(
    p_cal: float,
    *,
    low_upper: float,
    high_lower: float,
) -> str:
    if p_cal >= high_lower:
        return "high"
    if p_cal <= low_upper:
        return "low"
    return "medium"
