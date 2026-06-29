"""Paridad de resize previo a 224 (API staging)."""

from __future__ import annotations

from io import BytesIO

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


def test_prepare_prediction_image_keeps_full_rgb_for_inference() -> None:
    from PIL import Image

    from backend.inference.prediction_image_input import (
        encode_image_for_storage,
        validate_and_decode_prediction_image,
    )

    buf = BytesIO()
    Image.new("RGB", (2048, 1536), color=(180, 140, 120)).save(buf, format="PNG")
    raw = buf.getvalue()
    rgb = validate_and_decode_prediction_image("image/png", raw)
    assert rgb.shape[0] == 1536
    assert rgb.shape[1] == 2048
    _ct, stored = encode_image_for_storage(rgb)
    assert len(stored) < len(raw)
    stored_rgb = np.asarray(Image.open(BytesIO(stored)).convert("RGB"))
    assert max(stored_rgb.shape[0], stored_rgb.shape[1]) <= 1024
