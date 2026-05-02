"""
Heurística previa a la CNN: región tipo piel / uña en el encuadre.

No sustituye un detector dedicado; rechaza fondos uniformes u oscuros sin señal útil.
"""

from __future__ import annotations

import numpy as np

from backend.core.config import settings
from backend.core.exceptions import PredictionServiceError


def require_fingernail_presence(rgb_uint8: np.ndarray) -> None:
    """
    Comprueba que haya suficientes píxeles tipo piel (proxy de uña/dedo).

    ``rgb_uint8``: HWC, uint8, RGB.
    """
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise PredictionServiceError(
            "El archivo no es una imagen válida o está dañado.",
            400,
            code="image_not_decodable",
        )
    h, w, _ = rgb_uint8.shape
    if h < 8 or w < 8:
        raise PredictionServiceError(
            "La imagen es demasiado pequeña para evaluar la uña.",
            400,
            code="image_too_small",
        )
    r = rgb_uint8[:, :, 0].astype(np.float32)
    g = rgb_uint8[:, :, 1].astype(np.float32)
    b = rgb_uint8[:, :, 2].astype(np.float32)
    # Reglas simplificadas tipo RGB skin (adaptado de literatura clásica)
    skin = (
        (r > 95)
        & (g > 40)
        & (b > 20)
        & ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 15)
        & (np.abs(r - g) > 15)
        & (r > b)
        & (r > g)
    )
    ratio = float(np.mean(skin))
    if ratio < settings.nail_presence_min_skin_ratio:
        raise PredictionServiceError(
            "No se detecta una uña o dedo suficiente en la imagen.",
            400,
            code="no_fingernail_detected",
        )
