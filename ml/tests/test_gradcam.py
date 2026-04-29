"""Tests Grad-CAM (G10)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf
from tensorflow import keras

from ml.explainability.gradcam import GradCAM, GradCAMError


def _rgb_noise(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)


def test_gradcam_shapes_and_nonzero_heatmap(keras_model_path) -> None:
    tf.keras.utils.set_random_seed(42)
    model = keras.models.load_model(keras_model_path, compile=False)
    gc = GradCAM(model)
    rgb = _rgb_noise(1)
    res = gc.explain(rgb)
    assert res.heatmap.shape == (224, 224)
    assert res.overlay.shape == (224, 224, 3)
    assert res.saliency.shape == (224, 224)
    assert res.preprocessed.shape == (1, 224, 224, 3)
    assert float(np.max(res.heatmap)) > 1e-6


def test_gradcam_deterministic(keras_model_path) -> None:
    tf.keras.utils.set_random_seed(7)
    model = keras.models.load_model(keras_model_path, compile=False)
    gc = GradCAM(model)
    rgb = _rgb_noise(2)
    a = gc.explain(rgb).heatmap
    tf.keras.utils.set_random_seed(7)
    model2 = keras.models.load_model(keras_model_path, compile=False)
    gc2 = GradCAM(model2)
    b = gc2.explain(rgb).heatmap
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)


def test_explicit_layer_override(keras_model_path) -> None:
    model = keras.models.load_model(keras_model_path, compile=False)
    backbone = model.get_layer("mobilenet_backbone")
    first_dw = next(
        lyr
        for lyr in backbone.layers
        if isinstance(lyr, keras.layers.DepthwiseConv2D)
    )
    gc = GradCAM(model, layer_name=first_dw.name)
    assert gc.selected_layer == first_dw.name
    res = gc.explain(_rgb_noise(3))
    assert res.heatmap.shape == (224, 224)


def test_invalid_layer_raises(keras_model_path) -> None:
    model = keras.models.load_model(keras_model_path, compile=False)
    with pytest.raises(GradCAMError):
        GradCAM(model, layer_name="this_layer_does_not_exist_12345")
