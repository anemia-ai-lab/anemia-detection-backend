"""Paridad numérica Keras vs TFLite (G8)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf

from backend.inference.keras_image_predictor import KerasImagePredictor
from backend.inference.probability_calibration import (
    apply_temperature_calibration,
    binary_prediction_from_threshold,
)
from ml.inference.tflite_inference import TFLiteInferenceEngine
from ml.inference.tflite_inference import get_tflite_engine as _get_cached_engine
from ml.preprocessing.pipeline import PreprocessingConfig


def test_raw_and_calibrated_parity(
    keras_model_path,
    tflite_paths: tuple,
) -> None:
    _get_cached_engine.cache_clear()
    tf.keras.utils.set_random_seed(2026)
    rng = np.random.default_rng(2026)
    arr = rng.integers(0, 256, size=(120, 140, 3), dtype=np.uint8)
    raw_bytes = tf.io.encode_png(tf.constant(arr)).numpy()
    raw_bytes = bytes(raw_bytes)

    pre_cfg = PreprocessingConfig()
    keras_p = KerasImagePredictor(keras_model_path)
    raw_k = float(keras_p.predict_score(raw_bytes))

    tfl, meta = tflite_paths
    eng = TFLiteInferenceEngine(tfl, meta, preprocess_cfg=pre_cfg)
    r = eng.predict(raw_bytes)

    raw_delta = abs(raw_k - r.raw_probability)
    assert raw_delta < 1e-5, (
        "Paridad raw Keras vs TFLite: |Δ| debe ser < 1e-5. "
        f"keras={raw_k!r} tflite={r.raw_probability!r} |Δ|={raw_delta!r}. "
        "Causas habituales: .tflite exportado con optimizaciones/cuantificación distintas al "
        "script actual (float32, sin ``Optimize.DEFAULT``), versión de TensorFlow distinta al "
        "exportar, o preprocesado distinto (mismo ``PreprocessingConfig`` en ambos caminos)."
    )

    cal_k = apply_temperature_calibration(raw_k, r.temperature)
    cal_delta = abs(r.calibrated_probability - cal_k)
    assert cal_delta < 1e-6, (
        "Paridad calibrada: misma temperatura y fórmula que el backend; |Δ| debe ser < 1e-6. "
        f"cal_keras={cal_k!r} cal_tflite={r.calibrated_probability!r} |Δ|={cal_delta!r} "
        f"(raw_keras={raw_k!r}, raw_tflite={r.raw_probability!r})."
    )

    th = float(r.threshold_used)
    pred_k = int(binary_prediction_from_threshold(cal_k, th))
    assert pred_k == r.prediction, (
        "La predicción binaria debe coincidir usando el umbral operacional sobre la probabilidad "
        f"calibrada a partir del raw Keras. pred_keras={pred_k} pred_tflite={r.prediction} "
        f"threshold={th!r} cal_keras={cal_k!r} cal_tflite={r.calibrated_probability!r}."
    )
