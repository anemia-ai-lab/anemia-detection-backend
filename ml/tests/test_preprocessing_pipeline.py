"""Tests del pipeline G9."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from ml.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingError,
    preprocess_image_bytes,
    preprocess_rgb_array,
)


def test_preprocess_rgb_shape_and_dtype() -> None:
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    r = preprocess_rgb_array(rgb)
    assert r.model_input_tensor.shape == (1, 224, 224, 3)
    assert r.model_input_tensor.dtype == np.float32
    assert r.decoded_rgb_uint8.shape == (48, 64, 3)


def test_preprocess_deterministic_same_input() -> None:
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    a = preprocess_rgb_array(rgb).model_input_tensor
    b = preprocess_rgb_array(rgb).model_input_tensor
    np.testing.assert_array_equal(a, b)


def test_parity_with_legacy_mobilenet_stack() -> None:
    """Mismo orden que ``KerasImagePredictor`` histórico / ``dataops`` (sin iluminación extra)."""
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(90, 110, 3), dtype=np.uint8)
    got = preprocess_rgb_array(rgb, cfg=PreprocessingConfig()).model_input_tensor

    t = tf.constant(rgb)
    ref = tf.image.resize(t, [224, 224])
    ref = tf.cast(ref, tf.float32)
    ref = preprocess_input(ref)
    ref_b = tf.expand_dims(ref, 0).numpy()

    np.testing.assert_allclose(got, ref_b, rtol=1e-5, atol=1e-5)


def test_invalid_too_small() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(PreprocessingError):
        preprocess_rgb_array(rgb)


def test_invalid_bad_dtype() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.float32)
    with pytest.raises(PreprocessingError):
        preprocess_rgb_array(rgb)


def test_empty_bytes() -> None:
    with pytest.raises(PreprocessingError):
        preprocess_image_bytes(b"")


def test_lighting_mode_changes_tensor() -> None:
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    base = preprocess_rgb_array(
        rgb,
        cfg=PreprocessingConfig(lighting_normalization="off"),
    ).model_input_tensor
    lit = preprocess_rgb_array(
        rgb,
        cfg=PreprocessingConfig(lighting_normalization="brightness_contrast_stabilize"),
    ).model_input_tensor
    assert not np.allclose(base, lit)


def test_roi_hook_called() -> None:
    called: dict[str, bool] = {"ok": False}

    def hook(x: np.ndarray) -> np.ndarray:
        called["ok"] = True
        return x

    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
    preprocess_rgb_array(rgb, cfg=PreprocessingConfig(roi_extractor=hook))
    assert called["ok"] is True
