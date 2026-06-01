"""Paridad de resize previo a 224 (API staging)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from backend.inference.prediction_image_input import resize_rgb_max_edge


def _large_rgb(w: int, h: int) -> np.ndarray:
    img = Image.new("RGB", (w, h), color=(180, 140, 120))
    return np.asarray(img, dtype=np.uint8)


def test_resize_rgb_max_edge_scales_down_large_image() -> None:
    rgb = _large_rgb(2048, 1536)
    out = resize_rgb_max_edge(rgb, max_edge=1024)
    assert max(out.shape[0], out.shape[1]) <= 1024
    assert out.dtype == np.uint8


def test_resize_rgb_max_edge_noop_when_already_small() -> None:
    rgb = _large_rgb(224, 224)
    out = resize_rgb_max_edge(rgb, max_edge=1024)
    assert out.shape == rgb.shape
