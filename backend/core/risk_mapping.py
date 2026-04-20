from typing import Literal

RiskLevel = Literal["low", "high"]


def risk_from_probability(score: float, threshold: float) -> RiskLevel:
    """Mapa binario v1: ``score`` como probabilidad en [0, 1]."""
    if score >= threshold:
        return "high"
    return "low"
