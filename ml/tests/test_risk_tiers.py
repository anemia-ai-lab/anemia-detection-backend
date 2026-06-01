import numpy as np

from baseline.risk_tiers import risk_tier_from_probability, risk_tier_thresholds_from_validation


def test_risk_tier_thresholds_ordering() -> None:
    y = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8], dtype=np.float64)
    tiers = risk_tier_thresholds_from_validation(y, p)
    assert tiers["low_upper"] < tiers["high_lower"]


def test_risk_tier_from_probability_bands() -> None:
    assert risk_tier_from_probability(0.05, low_upper=0.2, high_lower=0.5) == "low"
    assert risk_tier_from_probability(0.35, low_upper=0.2, high_lower=0.5) == "medium"
    assert risk_tier_from_probability(0.6, low_upper=0.2, high_lower=0.5) == "high"
