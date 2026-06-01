import pytest

from backend.core.risk_mapping import risk_from_probability, risk_from_probability_binary


@pytest.mark.parametrize(
    ("score", "low_upper", "high_lower", "expected"),
    [
        (0.05, 0.2, 0.5, "low"),
        (0.35, 0.2, 0.5, "medium"),
        (0.6, 0.2, 0.5, "high"),
        (0.42, 0.5, 0.5, "low"),
        (0.42, 0.1, 0.41, "high"),
    ],
)
def test_risk_from_probability_tiers(
    score: float,
    low_upper: float,
    high_lower: float,
    expected: str,
) -> None:
    assert risk_from_probability(score, low_upper=low_upper, high_lower=high_lower) == expected


@pytest.mark.parametrize(
    ("score", "threshold", "expected"),
    [
        (0.49, 0.5, "low"),
        (0.5, 0.5, "high"),
    ],
)
def test_risk_from_probability_binary(score: float, threshold: float, expected: str) -> None:
    assert risk_from_probability_binary(score, threshold) == expected
