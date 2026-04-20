"""Contrato mínimo para obtener un score de probabilidad desde bytes de imagen."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ImagePredictor(Protocol):
    def predict_score(self, image_bytes: bytes) -> float:
        """Probabilidad en [0, 1] (clase positiva)."""
        ...


class StaticImagePredictor:
    """Predictor fijo para tests o entornos sin TensorFlow."""

    def __init__(self, score: float) -> None:
        self._score = score

    def predict_score(self, image_bytes: bytes) -> float:
        _ = image_bytes
        return self._score
