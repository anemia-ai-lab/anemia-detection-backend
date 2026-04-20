"""Calibración post-hoc por temperatura (solo inferencia; el modelo .keras no cambia)."""

from __future__ import annotations

import math

# Evita log(0) y sigmoid con exponentes extremos.
_EPS_PROB: float = 1e-7
_LOGIT_CLIP: float = 20.0


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
    p = float(raw_probability)
    p = min(max(p, eps), 1.0 - eps)
    logit_p = math.log(p / (1.0 - p))
    logit_p = max(-_LOGIT_CLIP, min(_LOGIT_CLIP, logit_p))
    z = logit_p / T
    z = max(-_LOGIT_CLIP, min(_LOGIT_CLIP, z))
    return float(1.0 / (1.0 + math.exp(-z)))


def binary_prediction_from_threshold(calibrated_probability: float, threshold: float) -> int:
    """Clase positiva (1) si la probabilidad calibrada supera el umbral operacional."""
    return 1 if float(calibrated_probability) >= float(threshold) else 0
