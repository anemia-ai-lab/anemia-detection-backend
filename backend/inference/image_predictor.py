"""Contrato mínimo para obtener un score de probabilidad desde bytes de imagen."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class ImagePredictor(Protocol):
    def predict_score(self, image_bytes: bytes) -> float:
        """Probabilidad en [0, 1] (clase positiva)."""
        pass

    def predict_from_rgb(self, rgb_uint8: "np.ndarray") -> float:
        """Probabilidad desde RGB uint8 HWC (sin re-decodificar PNG tras preparación)."""
        pass


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


class SequenceImagePredictor:
    """Devuelve scores en secuencia por cada crop (tests multinail)."""

    def __init__(self, scores: list[float]) -> None:
        if not scores:
            msg = "SequenceImagePredictor requiere al menos un score"
            raise ValueError(msg)
        self._scores = scores
        self._index = 0

    def predict_from_rgb(self, rgb_uint8: object) -> float:
        _ = rgb_uint8
        idx = min(self._index, len(self._scores) - 1)
        self._index += 1
        return self._scores[idx]

    def predict_score(self, image_bytes: bytes) -> float:
        return self.predict_from_rgb(image_bytes)
