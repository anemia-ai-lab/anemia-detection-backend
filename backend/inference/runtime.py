"""Carga única del modelo Keras al arranque (opcional según configuración)."""

from __future__ import annotations

import logging
from pathlib import Path

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
        logger.info("INFERENCE_MODEL_PATH vacío: sin modelo Keras en arranque.")
        return
    if not path.is_file():
        logger.warning("INFERENCE_MODEL_PATH no es un fichero existente: %s", path)
        return
    _builtin_predictor = KerasImagePredictor(path)
    logger.info("Modelo Keras cargado para inferencia: %s", path)


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
