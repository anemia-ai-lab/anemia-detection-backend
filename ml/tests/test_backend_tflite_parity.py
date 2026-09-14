"""Paridad numérica Keras vs TFLite (G8)."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf

from backend.inference.keras_image_predictor import KerasImagePredictor
from backend.inference.probability_calibration import (
    apply_probability_calibration,
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

    pre_cfg = PreprocessingConfig()
    keras_p = KerasImagePredictor(keras_model_path)
    raw_k = float(keras_p.predict_from_rgb(arr))

    tfl, meta = tflite_paths
    eng = TFLiteInferenceEngine(tfl, meta, preprocess_cfg=pre_cfg)
    r = eng.predict_rgb(arr)

    raw_delta = abs(raw_k - r.raw_probability)
    assert raw_delta < 1e-5, (
        "Paridad raw Keras vs TFLite: |Δ| debe ser < 1e-5. "
        f"keras={raw_k!r} tflite={r.raw_probability!r} |Δ|={raw_delta!r}. "
        "Causas habituales: .tflite exportado con optimizaciones/cuantificación distintas al "
        "script actual (float32, sin ``Optimize.DEFAULT``), versión de TensorFlow distinta al "
        "exportar, o preprocesado distinto (mismo ``PreprocessingConfig`` en ambos caminos)."
    )

    meta_data = json.loads(meta.read_text(encoding="utf-8"))
    cal_k = apply_probability_calibration(
        raw_k,
        method=str(meta_data.get("calibration_method") or "temperature"),
        temperature=float(meta_data["temperature"]),
        platt_a=float(meta_data.get("platt_a", 1.0)),
        platt_b=float(meta_data.get("platt_b", 0.0)),
    )
    cal_delta = abs(r.calibrated_probability - cal_k)
    assert cal_delta < 1e-5, (
        "Paridad calibrada: misma temperatura y fórmula que el backend; |Δ| debe ser < 1e-5. "
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
