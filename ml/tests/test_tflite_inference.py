"""Tests de inferencia TFLite (G8)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from ml.inference.tflite_inference import (
    TFLiteExportMetadata,
    TFLiteInferenceEngine,
    TFLiteMetadataError,
    get_tflite_engine,
)
from ml.preprocessing.pipeline import PreprocessingConfig


def test_metadata_parse(tflite_paths: tuple[Path, Path]) -> None:
    _, meta = tflite_paths
    data = json.loads(meta.read_text(encoding="utf-8"))
    m = TFLiteExportMetadata.from_json_dict(data)
    assert m.model_version == "v1.0"
    assert m.temperature > 0


def test_metadata_rejects_bad_flags(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(
        json.dumps(
            {
                "model_version": "x",
                "temperature": 1.0,
                "operational_threshold": 0.5,
                "raw_output_is_sigmoid_probability": False,
                "temperature_scaling_applied_inside_graph": False,
            },
        ),
        encoding="utf-8",
    )
    with pytest.raises(TFLiteMetadataError):
        TFLiteExportMetadata.from_json_dict(json.loads(p.read_text(encoding="utf-8")))


def test_engine_lazy_load_predict(tflite_paths: tuple[Path, Path]) -> None:
    import tensorflow as tf

    tfl, meta = tflite_paths
    get_tflite_engine.cache_clear()
    eng = TFLiteInferenceEngine(tfl, meta, preprocess_cfg=PreprocessingConfig())
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :] = (200, 150, 120)
    raw = tf.io.encode_png(tf.constant(arr)).numpy()
    r = eng.predict(bytes(raw))
    assert 0.0 <= r.raw_probability <= 1.0
    assert 0.0 <= r.calibrated_probability <= 1.0
    assert r.prediction in (0, 1)
    assert r.inference_mode == "tflite_offline"
    assert "raw_bytes_sha256" in (r.preprocessing or {})


def test_batch_order_stable(tflite_paths: tuple[Path, Path]) -> None:
    import tensorflow as tf

    tfl, meta = tflite_paths
    eng = TFLiteInferenceEngine(tfl, meta)
    images = []
    for c in ((100, 50, 30), (200, 180, 160)):
        arr = np.zeros((48, 48, 3), dtype=np.uint8)
        arr[:, :] = c
        images.append(bytes(tf.io.encode_png(tf.constant(arr)).numpy()))
    out = eng.predict_batch(images)
    assert len(out) == 2


def test_predict_deterministic(tflite_paths: tuple[Path, Path]) -> None:
    import tensorflow as tf

    tfl, meta = tflite_paths
    eng = TFLiteInferenceEngine(tfl, meta)
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :] = (180, 100, 90)
    raw = bytes(tf.io.encode_png(tf.constant(arr)).numpy())
    a = eng.predict(raw)
    b = eng.predict(raw)
    assert a.raw_probability == pytest.approx(b.raw_probability, rel=0, abs=1e-9)
    assert a.calibrated_probability == pytest.approx(b.calibrated_probability, rel=0, abs=1e-9)
