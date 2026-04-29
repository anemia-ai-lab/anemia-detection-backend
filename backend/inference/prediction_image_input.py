"""Validación y preparación de imagen antes de la heurística de uña y la CNN."""

from __future__ import annotations

import numpy as np

from backend.core.config import settings
from backend.core.prediction_image_limits import (
    ALLOWED_IMAGE_CONTENT_TYPES,
    prediction_image_max_bytes,
)
from backend.services.exceptions import PredictionServiceError

# Tras decodificar, se normaliza a PNG (uña, CNN y Storage).
_OUTPUT_CONTENT_TYPE = "image/png"


def decode_rgb_uint8(raw: bytes) -> np.ndarray:
    """Decodifica a RGB uint8 HWC."""
    try:
        import tensorflow as tf

        t = tf.io.decode_image(raw, channels=3, expand_animations=False)
        t.set_shape([None, None, 3])
        return tf.cast(t, tf.uint8).numpy()
    except Exception as exc:
        raise PredictionServiceError(
            "El archivo no es una imagen válida o está dañado.",
            400,
            code="image_not_decodable",
        ) from exc


def resize_rgb_max_edge(rgb: np.ndarray, max_edge: int) -> np.ndarray:
    """Escala manteniendo aspecto si el lado mayor supera ``max_edge``."""
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    m = max(h, w)
    if m <= max_edge:
        return rgb
    import tensorflow as tf

    scale = max_edge / m
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    t = tf.constant(rgb)
    out = tf.image.resize(t, [nh, nw], method=tf.image.ResizeMethod.BILINEAR)
    return tf.cast(tf.clip_by_value(tf.round(out), 0, 255), tf.uint8).numpy()


def require_rgb_pixel_limit(rgb: np.ndarray, max_pixels: int) -> None:
    """Evita expandir imágenes comprimidas a tensores enormes antes del resize."""
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    pixels = h * w
    if pixels > max_pixels:
        mp = max_pixels / 1_000_000
        raise PredictionServiceError(
            f"La imagen supera la resolución máxima permitida ({mp:.1f} MP).",
            413,
            code="image_resolution_too_large",
        )


def rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    import tensorflow as tf

    t = tf.constant(rgb)
    return bytes(tf.image.encode_png(t).numpy())


def prepare_prediction_image(content_type: str | None, raw: bytes) -> tuple[str, bytes, np.ndarray]:
    """
    Comprueba tamaño y MIME, decodifica, redimensiona y devuelve PNG + RGB para la uña.

    Devuelve ``(content_type_png, png_bytes, rgb_uint8_resized)``.
    """
    max_b = prediction_image_max_bytes()
    if len(raw) > max_b:
        mb = max_b / (1024 * 1024)
        raise PredictionServiceError(
            f"La imagen supera el tamaño máximo permitido ({mb:.0f} MB).",
            413,
            code="image_too_large",
        )
    raw_ct = (content_type or "application/octet-stream").split(";")[0].strip().lower()
    if raw_ct not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise PredictionServiceError(
            "El tipo de imagen no está permitido. Use JPEG, PNG o WebP.",
            415,
            code="unsupported_media_type",
        )
    rgb = decode_rgb_uint8(raw)
    require_rgb_pixel_limit(rgb, settings.prediction_image_max_pixels)
    rgb = resize_rgb_max_edge(rgb, settings.prediction_image_max_edge_px)
    png_bytes = rgb_to_png_bytes(rgb)
    return _OUTPUT_CONTENT_TYPE, png_bytes, rgb
