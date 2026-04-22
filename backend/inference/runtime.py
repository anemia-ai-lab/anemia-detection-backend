"""Carga única del modelo Keras al arranque (opcional según configuración)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from backend.core.config import repo_root, settings
from backend.inference.keras_image_predictor import KerasImagePredictor

logger = logging.getLogger(__name__)

_builtin_predictor: KerasImagePredictor | None = None


def _resolved_model_path() -> Path | None:
    raw = settings.inference_model_path.strip()
    if raw == "":
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = repo_root() / p
    return p


def init_inference_model() -> None:
    """Invocado desde el lifespan de FastAPI."""
    global _builtin_predictor
    if _builtin_predictor is not None:
        return
    path = _resolved_model_path()
    if path is None:
        logger.info("inference_model_skip reason=no_path_configured")
        return
    if not path.is_file():
        logger.warning("inference_model_load_failed reason=file_not_found path=%s", path)
        return
    try:
        _builtin_predictor = KerasImagePredictor(path)
    except Exception:
        logger.exception("inference_model_load_failed reason=keras_load_error path=%s", path)
        _builtin_predictor = None
        return
    logger.info("inference_model_loaded path=%s", path)


def inference_service_status() -> tuple[Literal["ok", "degraded"], bool]:
    """Estado de inferencia para ``/health`` (sin exponer secretos)."""
    path = _resolved_model_path()
    loaded = _builtin_predictor is not None
    if path is None:
        return "ok", loaded
    if not path.is_file():
        return "degraded", False
    if loaded:
        return "ok", True
    return "degraded", False


def shutdown_inference_model() -> None:
    global _builtin_predictor
    _builtin_predictor = None
    try:
        tf = __import__("tensorflow")
        tf.keras.backend.clear_session()
    except Exception:
        pass


def get_builtin_image_predictor() -> KerasImagePredictor | None:
    return _builtin_predictor
