"""Calibración post-hoc en inferencia (el modelo .keras no cambia).

Métodos soportados:
- ``temperature``: ``sigmoid(logit(p) / T)``
- ``platt``: ``sigmoid(a * logit(p) + b)``  (identidad: a=1, b=0)
"""

from __future__ import annotations

import math
from typing import Literal

CalibrationMethod = Literal["temperature", "platt"]

# Evita log(0) y sigmoid con exponentes extremos.
_EPS_PROB: float = 1e-7
_LOGIT_CLIP: float = 20.0


def _stable_logit(raw_probability: float, eps: float) -> float:
    p = min(max(float(raw_probability), eps), 1.0 - eps)
    logit_p = math.log(p / (1.0 - p))
    return max(-_LOGIT_CLIP, min(_LOGIT_CLIP, logit_p))


def _stable_sigmoid(z: float) -> float:
    z = max(-_LOGIT_CLIP, min(_LOGIT_CLIP, float(z)))
    return float(1.0 / (1.0 + math.exp(-z)))


def apply_temperature_calibration(
    raw_probability: float,
    temperature: float,
    *,
    eps: float = _EPS_PROB,
) -> float:
    """
    ``calibrated_p = sigmoid(logit(p) / T)`` con recorte numérico estable.

    Args:
        raw_probability: salida sigmoide del modelo en ``(0, 1)``.
        temperature: ``T > 0`` (p. ej. ajustado en validación con *temperature scaling*).
    """
    T = float(max(float(temperature), eps))
    return _stable_sigmoid(_stable_logit(raw_probability, eps) / T)


def apply_platt_calibration(
    raw_probability: float,
    platt_a: float,
    platt_b: float,
    *,
    eps: float = _EPS_PROB,
) -> float:
    """``calibrated_p = sigmoid(a * logit(p) + b)``. ``(a=1, b=0)`` deja ``p`` casi igual."""
    z = float(platt_a) * _stable_logit(raw_probability, eps) + float(platt_b)
    return _stable_sigmoid(z)


def apply_probability_calibration(
    raw_probability: float,
    *,
    method: CalibrationMethod | str = "temperature",
    temperature: float = 1.0,
    platt_a: float = 1.0,
    platt_b: float = 0.0,
) -> float:
    """Despacha temperature scaling o Platt. Método desconocido → temperature."""
    resolved = str(method).strip().lower()
    if resolved == "platt":
        return apply_platt_calibration(raw_probability, platt_a, platt_b)
    return apply_temperature_calibration(raw_probability, temperature)


def calibration_is_enabled(
    *,
    method: CalibrationMethod | str,
    temperature: float,
    platt_a: float,
    platt_b: float,
) -> bool:
    """True si la calibración no es la identidad numérica."""
    resolved = str(method).strip().lower()
    if resolved == "platt":
        return abs(float(platt_a) - 1.0) > 1e-12 or abs(float(platt_b)) > 1e-12
    return abs(float(temperature) - 1.0) > 1e-12


def binary_prediction_from_threshold(calibrated_probability: float, threshold: float) -> int:
    """Clase positiva (1) si la probabilidad calibrada supera el umbral operacional."""
    return 1 if float(calibrated_probability) >= float(threshold) else 0
