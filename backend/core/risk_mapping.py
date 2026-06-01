from typing import Literal

RiskLevel = Literal["low", "medium", "high"]


def risk_from_probability(
    score: float,
    *,
    low_upper: float,
    high_lower: float,
) -> RiskLevel:
    """Tres niveles sobre probabilidad calibrada (umbrales desde validación Ghana)."""
    if score >= high_lower:
        return "high"
    if score <= low_upper:
        return "low"
    return "medium"


def risk_from_probability_binary(score: float, threshold: float) -> Literal["low", "high"]:
    """Compatibilidad binaria cuando solo hay un umbral operacional."""
    if score >= threshold:
        return "high"
    return "low"


def anemia_risk_label(risk: RiskLevel) -> str:
    """Texto corto para demos (no sustituye los campos numéricos ni ``risk``)."""
    if risk == "high":
        return "High anemia risk prediction"
    if risk == "medium":
        return "Medium anemia risk prediction"
    return "Low anemia risk prediction"
