"""Carga MobileNetV2 baseline (.keras) y ``predict`` con preprocesado alineado al entrenamiento."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class KerasImagePredictor:
    """Resize 224×224 + ``mobilenet_v2.preprocess_input`` + ``model.predict``."""

    def __init__(self, model_path: Path) -> None:
        from tensorflow import keras

        self._model: Any = keras.models.load_model(model_path, compile=False)

    def predict_score(self, image_bytes: bytes) -> float:
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        batch = self._preprocess(image_bytes, tf=tf, preprocess_input=preprocess_input)
        out = self._model.predict(batch, verbose=0)
        score = float(np.asarray(out).squeeze())
        return max(0.0, min(1.0, score))

    @staticmethod
    def _preprocess(
        image_bytes: bytes,
        *,
        tf: Any,
        preprocess_input: Any,
    ) -> Any:
        img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return tf.expand_dims(img, 0)
