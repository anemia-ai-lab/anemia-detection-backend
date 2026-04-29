"""Carga MobileNetV2 baseline (.keras) y ``predict`` con preprocesado alineado al entrenamiento."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ml.preprocessing.pipeline import (
    PreprocessingConfig,
    preprocess_image_bytes,
    preprocess_rgb_array,
)


def _tf_disabled() -> bool:
    v = os.environ.get("DISABLE_TF", "").strip().lower()
    return v in ("1", "true", "yes", "on")


class KerasImagePredictor:
    """Pipeline G9 + ``model.predict`` (salida sigmoide escalar)."""

    def __init__(self, model_path: Path) -> None:
        if _tf_disabled():
            msg = "KerasImagePredictor no disponible con DISABLE_TF=1 (entornos de test sin TensorFlow)."
            raise RuntimeError(msg)
        from tensorflow import keras

        self._model: Any = keras.models.load_model(model_path, compile=False)
        self._pre_cfg = PreprocessingConfig()

    def predict_from_rgb(self, rgb_uint8: np.ndarray) -> float:
        """Inferencia desde RGB ya validado (sin re-decodificar el PNG intermedio)."""
        batch = preprocess_rgb_array(rgb_uint8, cfg=self._pre_cfg).model_input_tensor
        out = self._model.predict(batch, verbose=0)
        score = float(np.asarray(out).squeeze())
        return max(0.0, min(1.0, score))

    def predict_score(self, image_bytes: bytes) -> float:
        """Compatibilidad: decodifica bytes y aplica el mismo pipeline G9."""
        batch = preprocess_image_bytes(image_bytes, cfg=self._pre_cfg).model_input_tensor
        out = self._model.predict(batch, verbose=0)
        score = float(np.asarray(out).squeeze())
        return max(0.0, min(1.0, score))
