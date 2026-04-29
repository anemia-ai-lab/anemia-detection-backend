"""Contrato mínimo para obtener un score de probabilidad desde bytes de imagen."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class ImagePredictor(Protocol):
    def predict_score(self, image_bytes: bytes) -> float:
        """Probabilidad en [0, 1] (clase positiva)."""
        ...

    def predict_from_rgb(self, rgb_uint8: "np.ndarray") -> float:
        """Probabilidad desde RGB uint8 HWC (sin re-decodificar PNG tras preparación)."""
        ...


class StaticImagePredictor:
    """Predictor fijo para tests o entornos sin TensorFlow."""

    def __init__(self, score: float) -> None:
        self._score = score

    def predict_score(self, image_bytes: bytes) -> float:
        _ = image_bytes
        return self._score

    def predict_from_rgb(self, rgb_uint8: object) -> float:
        _ = rgb_uint8
        return self._score
