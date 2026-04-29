"""
Pipeline de preprocesado G9: único camino para backend Keras, TFLite offline y Grad-CAM.

Por defecto replica el flujo de entrenamiento (``ml/baseline/dataops.py``):
decode → resize 224 → float32 → ``mobilenet_v2.preprocess_input``.

La normalización de iluminación es **opcional** y viene **desactivada** para conservar
paridad con la calibración publicada (temperature / umbral operacional).
"""

from __future__ import annotations

import hashlib
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

_ML_ROOT = Path(__file__).resolve().parents[1]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from baseline.config import IMG_SIZE, SEED

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "g9-v1"

LightingMode = Literal["off", "per_image_standardize", "brightness_contrast_stabilize"]
ResizeMethod = Literal["bilinear"]


class PreprocessingError(ValueError):
    """Entrada de imagen inválida o no soportada para el pipeline."""


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuración reproducible del pipeline (sin rutas mágicas en el código de inferencia)."""

    input_size: tuple[int, int] = IMG_SIZE
    resize_method: ResizeMethod = "bilinear"
    lighting_normalization: LightingMode = "off"
    roi_extractor: Callable[[np.ndarray], np.ndarray] | None = None
    deterministic_seed: int = SEED

    def __post_init__(self) -> None:
        h, w = self.input_size
        if h < 1 or w < 1:
            msg = f"input_size inválido: {self.input_size}"
            raise ValueError(msg)


@dataclass
class PreprocessingResult:
    """Salida del pipeline: tensor de modelo + RGB de referencia + trazabilidad."""

    decoded_rgb_uint8: np.ndarray
    model_input_tensor: np.ndarray
    provenance: dict[str, Any] = field(default_factory=dict)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decode_rgb_uint8(raw: bytes) -> np.ndarray:
    import tensorflow as tf

    if not raw:
        raise PreprocessingError("bytes de imagen vacíos")
    try:
        t = tf.io.decode_image(raw, channels=3, expand_animations=False)
        t.set_shape([None, None, 3])
        return tf.cast(t, tf.uint8).numpy()
    except Exception as exc:
        raise PreprocessingError("no se pudo decodificar la imagen") from exc


def _validate_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise PreprocessingError("se esperaba imagen HWC con 3 canales")
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if h < 8 or w < 8:
        raise PreprocessingError("imagen demasiado pequeña (mínimo 8×8)")
    if rgb.dtype != np.uint8:
        raise PreprocessingError("se esperaba dtype uint8 para RGB de entrada")
    if not np.isfinite(rgb.astype(np.float32)).all():
        raise PreprocessingError("valores no finitos en la imagen")


def _resize_to_model_input(
    rgb_uint8: np.ndarray,
    *,
    input_size: tuple[int, int],
    resize_method: ResizeMethod,
) -> np.ndarray:
    import tensorflow as tf

    method = tf.image.ResizeMethod.BILINEAR
    t = tf.constant(rgb_uint8)
    out = tf.image.resize(t, list(input_size), method=method)
    return tf.cast(out, tf.float32).numpy()


def _apply_lighting_normalization(
    img_f32_hwc: np.ndarray,
    mode: LightingMode,
    *,
    seed: int,
) -> np.ndarray:
    """Iluminación en float32 aprox. [0, 255]. Solo si mode != 'off'."""
    if mode == "off":
        return img_f32_hwc

    import tensorflow as tf

    t = tf.constant(img_f32_hwc, dtype=tf.float32)

    if mode == "per_image_standardize":
        # per_image_standardization espera float; re-escalamos a [0,255] para MobileNet.
        std = tf.image.per_image_standardization(t)
        std_np = std.numpy()
        lo = float(np.min(std_np))
        hi = float(np.max(std_np))
        if hi - lo < 1e-6:
            return img_f32_hwc
        scaled = (std_np - lo) / (hi - lo) * 255.0
        return np.clip(scaled, 0.0, 255.0).astype(np.float32)

    if mode == "brightness_contrast_stabilize":
        _ = seed  # API estable para futuras variantes estocásticas documentadas
        x = img_f32_hwc.astype(np.float64)
        out = np.empty_like(x, dtype=np.float32)
        for c in range(3):
            ch = x[:, :, c].ravel()
            p_lo, p_hi = np.percentile(ch, [1.0, 99.0])
            if p_hi - p_lo < 1e-6:
                out[:, :, c] = img_f32_hwc[:, :, c]
                continue
            y = (x[:, :, c] - p_lo) / (p_hi - p_lo) * 255.0
            out[:, :, c] = np.clip(y, 0.0, 255.0).astype(np.float32)
        return out

    msg = f"modo de iluminación desconocido: {mode}"
    raise PreprocessingError(msg)


def _apply_roi_hook(
    img_f32_hwc: np.ndarray,
    roi_extractor: Callable[[np.ndarray], np.ndarray] | None,
) -> np.ndarray:
    if roi_extractor is None:
        return img_f32_hwc
    out = roi_extractor(img_f32_hwc)
    if out.shape != img_f32_hwc.shape:
        raise PreprocessingError("roi_extractor debe devolver tensor de la misma forma HWC")
    if out.dtype != np.float32:
        out = out.astype(np.float32)
    return out


def _apply_mobilenet_preprocess(img_f32_hwc: np.ndarray) -> np.ndarray:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    return preprocess_input(img_f32_hwc).astype(np.float32)


def _add_batch_dim(chw: np.ndarray) -> np.ndarray:
    return np.expand_dims(chw, axis=0)


def preprocess_rgb_array(
    rgb_uint8: np.ndarray,
    *,
    cfg: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """
    Preprocesa RGB uint8 HWC (p. ej. tras ``prepare_prediction_image``) al tensor de modelo.

    Orden: validar, resize 224, iluminación opc., ROI hook, MobileNet preprocess, batch.
    """
    cfg = cfg or PreprocessingConfig()
    _validate_rgb(rgb_uint8)
    decoded_rgb_uint8 = np.ascontiguousarray(rgb_uint8)
    input_fp = _sha256_bytes(decoded_rgb_uint8.tobytes())

    resized = _resize_to_model_input(
        decoded_rgb_uint8,
        input_size=cfg.input_size,
        resize_method=cfg.resize_method,
    )
    after_light = _apply_lighting_normalization(
        resized,
        cfg.lighting_normalization,
        seed=cfg.deterministic_seed,
    )
    after_roi = _apply_roi_hook(after_light, cfg.roi_extractor)
    mobilenet_ready = _apply_mobilenet_preprocess(after_roi)
    batch = _add_batch_dim(mobilenet_ready)

    provenance: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "input_size": f"{cfg.input_size[0]}x{cfg.input_size[1]}",
        "resize_method": cfg.resize_method,
        "lighting_normalization": cfg.lighting_normalization,
        "mobilenet_preprocess": "mobilenet_v2.preprocess_input",
        "input_sha256": input_fp,
        "roi_hook_applied": cfg.roi_extractor is not None,
    }
    logger.debug("preprocess_rgb_array provenance=%s", provenance)

    return PreprocessingResult(
        decoded_rgb_uint8=decoded_rgb_uint8,
        model_input_tensor=batch,
        provenance=provenance,
    )


def preprocess_image_bytes(
    raw: bytes,
    *,
    cfg: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Decodifica bytes de imagen y aplica el mismo pipeline que ``preprocess_rgb_array``."""
    cfg = cfg or PreprocessingConfig()
    rgb = _decode_rgb_uint8(raw)
    res = preprocess_rgb_array(rgb, cfg=cfg)
    raw_hash = _sha256_bytes(raw)
    res.provenance["raw_bytes_sha256"] = raw_hash
    res.provenance["source"] = "image_bytes"
    return res
