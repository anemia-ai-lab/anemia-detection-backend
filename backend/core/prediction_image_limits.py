"""Límites y tipos permitidos para imágenes de predicción (compartido con Storage)."""

from __future__ import annotations

from typing import Final

from backend.core.config import settings

ALLOWED_IMAGE_CONTENT_TYPES: Final[frozenset[str]] = frozenset(
    {"image/jpeg", "image/png", "image/webp"},
)
IMAGE_EXT_BY_CONTENT_TYPE: Final[dict[str, str]] = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


def prediction_image_max_bytes() -> int:
    return settings.prediction_image_max_bytes
