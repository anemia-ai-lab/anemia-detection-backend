import pytest

from backend.core.risk_mapping import risk_from_probability


@pytest.mark.parametrize(
    ("score", "threshold", "expected"),
    [
        (0.0, 0.5, "low"),
        (0.49, 0.5, "low"),
        (0.5, 0.5, "high"),
        (1.0, 0.5, "high"),
        (0.42, 0.5, "low"),
        (0.42, 0.41, "high"),
    ],
)
def test_risk_from_probability(score: float, threshold: float, expected: str) -> None:
    assert risk_from_probability(score, threshold) == expected
