"""Promedio de probabilidades raw de varios checkpoints .keras."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.inference.keras_image_predictor import KerasImagePredictor


class EnsembleKerasImagePredictor:
    """Pipeline idéntico por miembro; devuelve media de ``raw_prob``."""

    def __init__(self, model_paths: list[Path]) -> None:
        if not model_paths:
            msg = "EnsembleKerasImagePredictor requiere al menos una ruta .keras"
            raise ValueError(msg)
        self._members = [KerasImagePredictor(p) for p in model_paths]

    def predict_from_rgb(self, rgb_uint8: np.ndarray) -> float:
        scores = [m.predict_from_rgb(rgb_uint8) for m in self._members]
        return float(np.mean(scores))

    def predict_score(self, image_bytes: bytes) -> float:
        scores = [m.predict_score(image_bytes) for m in self._members]
        return float(np.mean(scores))
