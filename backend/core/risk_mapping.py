from typing import Literal

RiskLevel = Literal["low", "high"]


def risk_from_probability(score: float, threshold: float) -> RiskLevel:
    """Mapa binario v1: ``score`` como probabilidad en [0, 1]."""
    if score >= threshold:
        return "high"
    return "low"


def anemia_risk_label(risk: RiskLevel) -> str:
    """Texto corto para demos (no sustituye los campos numéricos ni ``risk``)."""
    if risk == "high":
        return "High anemia risk prediction"
    return "Low anemia risk prediction"
