"""
Detección y recorte de uñas (índice, medio, anular) para POST /predict.

MediaPipe Hand Landmarker (Tasks API) localiza landmarks; OpenCV recorta la uña
rotada al eje tip→DIP (centro hacia el lecho, no en la yema).
"""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import cv2
import numpy as np

from backend.core.config import repo_root, settings
from backend.core.exceptions import PredictionServiceError

logger = logging.getLogger(__name__)

FingerLabel = Literal["index", "middle", "ring", "hand"]
CropSource = Literal["mediapipe", "fallback", "roi"]
FallbackMode = Literal["whole", "vertical_thirds", "reject"]

_SWAGGER_EMPTY_PLACEHOLDERS = frozenset({"string", "null", "undefined", "none"})

# Fracción del vector tip−DIP hacia el DIP: el lecho ungueal, no la yema.
_NAIL_BED_OFFSET = 0.35
# Etiqueta de preprocessing.crop (paridad online/offline).
CROP_PREPROCESS_LABEL = "tip_to_dip_rotated"

# MediaPipe hand landmark indices (21 points); misma topología que Hands legacy.
_FINGER_LANDMARKS: tuple[tuple[FingerLabel, int, int], ...] = (
    ("index", 8, 7),  # tip, DIP
    ("middle", 12, 11),
    ("ring", 16, 15),
)

_hand_landmarker_lock = threading.Lock()
_hand_landmarker: Any = None
_hand_landmarker_min_confidence: float | None = None


@dataclass(frozen=True)
class NailCrop:
    """Sub-imagen RGB de una uña lista para el pipeline G9."""

    finger: FingerLabel
    rgb: np.ndarray
    bbox: tuple[int, int, int, int]  # x, y, w, h en píxeles de la imagen fuente
    source: CropSource = "mediapipe"


@runtime_checkable
class NailDetector(Protocol):
    def detect(self, rgb_uint8: np.ndarray) -> list[NailCrop]:
        """Devuelve crops de uñas detectadas (puede ser vacío si fallback=reject)."""
        ...


def normalize_rois_input(raw: str | None) -> str | None:
    """Normaliza rois de formulario: vacío, whitespace y placeholders Swagger → omitido."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.lower() in _SWAGGER_EMPTY_PLACEHOLDERS:
        return None
    return s


def parse_roi_overrides(raw_json: str | None) -> list[dict[str, object]] | None:
    """Parsea JSON opcional de ROIs manuales (debug/ops)."""
    raw_json = normalize_rois_input(raw_json)
    if raw_json is None:
        return None
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise PredictionServiceError(
            "rois debe ser JSON válido.",
            422,
            code="invalid_rois_json",
        ) from exc
    if not isinstance(data, list):
        raise PredictionServiceError(
            "rois debe ser una lista de regiones.",
            422,
            code="invalid_rois_shape",
        )
    if len(data) == 0:
        return None
    if len(data) > 5:
        raise PredictionServiceError(
            "rois admite como máximo 5 regiones.",
            422,
            code="invalid_rois_shape",
        )
    return data


def _clamp_int(value: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(value))))


def _crop_axis_aligned(
    rgb: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    finger: FingerLabel,
    source: CropSource = "mediapipe",
) -> NailCrop | None:
    ih, iw = int(rgb.shape[0]), int(rgb.shape[1])
    x = _clamp_int(x, 0, iw - 1)
    y = _clamp_int(y, 0, ih - 1)
    w = max(8, min(w, iw - x))
    h = max(8, min(h, ih - y))
    patch = rgb[y : y + h, x : x + w].copy()
    if patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
        return None
    return NailCrop(finger=finger, rgb=patch, bbox=(x, y, w, h), source=source)


def crop_from_normalized_roi(
    rgb: np.ndarray,
    *,
    finger: FingerLabel,
    x: float,
    y: float,
    w: float,
    h: float,
) -> NailCrop | None:
    """Recorta región normalizada [0,1] relativa a la imagen."""
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        raise PredictionServiceError(
            "Cada ROI en rois debe tener x,y,w,h en [0,1] con w,h > 0.",
            422,
            code="invalid_rois_values",
        )
    ih, iw = int(rgb.shape[0]), int(rgb.shape[1])
    px = int(x * iw)
    py = int(y * ih)
    pw = max(8, int(w * iw))
    ph = max(8, int(h * ih))
    return _crop_axis_aligned(rgb, px, py, pw, ph, finger=finger, source="roi")


def _rotated_square_aabb(
    cx: float,
    cy: float,
    half: float,
    angle_deg: float,
    iw: int,
    ih: int,
) -> tuple[int, int, int, int]:
    """AABB en la imagen original del cuadrado axis-aligned tras rotar ``angle_deg`` alrededor de (cx, cy)."""
    rad = math.radians(angle_deg)
    alpha = math.cos(rad)
    beta = math.sin(rad)
    xs: list[float] = []
    ys: list[float] = []
    for lx, ly in ((-half, -half), (half, -half), (half, half), (-half, half)):
        # Inversa de cv2.getRotationMatrix2D (ángulo positivo = CCW).
        xs.append(alpha * lx - beta * ly + cx)
        ys.append(beta * lx + alpha * ly + cy)
    x0 = _clamp_int(min(xs), 0, iw - 1)
    y0 = _clamp_int(min(ys), 0, ih - 1)
    x1 = _clamp_int(max(xs), x0 + 8, iw)
    y1 = _clamp_int(max(ys), y0 + 8, ih)
    return (x0, y0, x1 - x0, y1 - y0)


def legacy_axis_aligned_tip_bbox(
    iw: int,
    ih: int,
    tip_xy: tuple[float, float],
    dip_xy: tuple[float, float],
    crop_scale: float,
) -> tuple[int, int, int, int]:
    """Bbox legado: cuadrado axis-aligned centrado en el tip (tests / comparación)."""
    tx, ty = tip_xy
    dx, dy = dip_xy
    finger_len = float(math.hypot(tx - dx, ty - dy))
    if finger_len < 1e-3:
        finger_len = min(ih, iw) * 0.05
    side = max(16.0, finger_len * 1.6 * crop_scale)
    half = side / 2.0
    x = _clamp_int(tx - half, 0, iw - 1)
    y = _clamp_int(ty - half, 0, ih - 1)
    x2 = _clamp_int(tx + half, x + 8, iw)
    y2 = _clamp_int(ty + half, y + 8, ih)
    return (x, y, x2 - x, y2 - y)


def crop_from_landmark_pair(
    rgb: np.ndarray,
    *,
    finger: FingerLabel,
    tip_xy: tuple[float, float],
    dip_xy: tuple[float, float],
    crop_scale: float,
) -> NailCrop | None:
    """Recorte cuadrado rotado al eje tip→DIP, con centro hacia el lecho ungueal.

    ``center = tip - 0.35 * (tip - DIP)``. Se rota para que DIP→tip quede vertical
    (punta hacia −Y). ``side = finger_len * 1.6 * crop_scale`` (default scale 1.0).
    El ``bbox`` es el AABB del cuadrado en coordenadas de la imagen original.
    """
    ih, iw = int(rgb.shape[0]), int(rgb.shape[1])
    tx, ty = float(tip_xy[0]), float(tip_xy[1])
    dx, dy = float(dip_xy[0]), float(dip_xy[1])
    vec_x = tx - dx
    vec_y = ty - dy
    axis_len = float(math.hypot(vec_x, vec_y))
    finger_len = axis_len if axis_len >= 1e-3 else min(ih, iw) * 0.05
    side = max(16.0, finger_len * 1.6 * crop_scale)
    out_size = max(16, int(round(side)))
    half = out_size / 2.0

    if axis_len < 1e-3:
        return _crop_axis_aligned(
            rgb,
            _clamp_int(tx - half, 0, iw - 1),
            _clamp_int(ty - half, 0, ih - 1),
            out_size,
            out_size,
            finger=finger,
        )

    cx = tx - _NAIL_BED_OFFSET * vec_x
    cy = ty - _NAIL_BED_OFFSET * vec_y
    current = math.atan2(vec_y, vec_x)
    desired = -math.pi / 2.0
    angle_deg = math.degrees(desired - current)

    matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    matrix[0, 2] += half - cx
    matrix[1, 2] += half - cy
    patch = cv2.warpAffine(
        rgb,
        matrix,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
        return None
    bbox = _rotated_square_aabb(cx, cy, half, angle_deg, iw, ih)
    return NailCrop(finger=finger, rgb=patch, bbox=bbox, source="mediapipe")


def _landmark_score(landmark: Any) -> float:
    visibility = getattr(landmark, "visibility", None)
    if visibility is not None:
        return float(visibility)
    presence = getattr(landmark, "presence", None)
    if presence is not None:
        return float(presence)
    return 1.0


def landmarks_to_crops(
    rgb_uint8: np.ndarray,
    landmarks: list[Any],
    *,
    min_confidence: float,
    crop_scale: float,
) -> list[NailCrop]:
    """Convierte 21 landmarks normalizados en crops de índice/medio/anular."""
    ih, iw = int(rgb_uint8.shape[0]), int(rgb_uint8.shape[1])
    crops: list[NailCrop] = []
    for finger, tip_idx, dip_idx in _FINGER_LANDMARKS:
        tip = landmarks[tip_idx]
        dip = landmarks[dip_idx]
        if _landmark_score(tip) < min_confidence or _landmark_score(dip) < min_confidence:
            continue
        tip_xy = (float(tip.x) * iw, float(tip.y) * ih)
        dip_xy = (float(dip.x) * iw, float(dip.y) * ih)
        crop = crop_from_landmark_pair(
            rgb_uint8,
            finger=finger,
            tip_xy=tip_xy,
            dip_xy=dip_xy,
            crop_scale=crop_scale,
        )
        if crop is not None:
            crops.append(crop)
    return crops


def resolved_hand_landmarker_model_path() -> Path:
    raw = settings.hand_landmarker_model_path.strip()
    p = Path(raw)
    if not p.is_absolute():
        p = repo_root() / p
    return p


_hand_landmarker_probe_cache: dict[str, object] | None = None


def get_cached_hand_landmarker_status(*, refresh: bool = False) -> dict[str, object]:
    """Estado cacheado del landmarker (startup / health)."""
    global _hand_landmarker_probe_cache
    if refresh or _hand_landmarker_probe_cache is None:
        _hand_landmarker_probe_cache = probe_hand_landmarker_status(warmup=True)
    return _hand_landmarker_probe_cache


def probe_hand_landmarker_status(*, warmup: bool = True) -> dict[str, object]:
    """
    Comprueba import de MediaPipe, existencia del .task y (opcional) carga del landmarker.

    Usado en startup y GET /health; no expone secretos.
    """
    model_path = resolved_hand_landmarker_model_path()
    status: dict[str, object] = {
        "ready": False,
        "model_path": str(model_path),
        "model_exists": model_path.is_file(),
        "import_ok": False,
        "error": None,
    }
    try:
        import mediapipe as mp  # noqa: F401

        status["import_ok"] = True
    except (ImportError, OSError) as exc:
        status["error"] = f"import_failed:{type(exc).__name__}:{exc}"
        return status

    if not model_path.is_file():
        status["error"] = "model_file_missing"
        return status

    if not warmup:
        status["ready"] = True
        return status

    try:
        _get_hand_landmarker(float(settings.predict_nail_detect_min_confidence))
        status["ready"] = True
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        status["error"] = f"init_failed:{type(exc).__name__}:{exc}"
    return status


def _get_hand_landmarker(min_confidence: float) -> Any:
    """Singleton thread-safe del detector Hand Landmarker (Tasks API)."""
    global _hand_landmarker, _hand_landmarker_min_confidence
    with _hand_landmarker_lock:
        if _hand_landmarker is not None and _hand_landmarker_min_confidence == min_confidence:
            return _hand_landmarker
        if _hand_landmarker is not None:
            _hand_landmarker.close()
            _hand_landmarker = None
            _hand_landmarker_min_confidence = None

        import mediapipe as mp

        model_path = resolved_hand_landmarker_model_path()
        if not model_path.is_file():
            msg = f"Modelo Hand Landmarker no encontrado: {model_path}"
            raise FileNotFoundError(msg)

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        _hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        _hand_landmarker_min_confidence = min_confidence
        return _hand_landmarker


def shutdown_hand_landmarker() -> None:
    """Libera el landmarker (p. ej. al apagar la app)."""
    global _hand_landmarker, _hand_landmarker_min_confidence
    with _hand_landmarker_lock:
        if _hand_landmarker is not None:
            _hand_landmarker.close()
            _hand_landmarker = None
            _hand_landmarker_min_confidence = None


def fallback_crops(rgb: np.ndarray, mode: FallbackMode) -> list[NailCrop]:
    if mode == "reject":
        return []
    if mode == "whole":
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        return [NailCrop(finger="hand", rgb=rgb.copy(), bbox=(0, 0, w, h), source="fallback")]
    # vertical_thirds
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    third = max(8, w // 3)
    labels: tuple[FingerLabel, ...] = ("index", "middle", "ring")
    crops: list[NailCrop] = []
    for i, finger in enumerate(labels):
        x = i * third
        cw = third if i < 2 else w - x
        crop = _crop_axis_aligned(rgb, x, 0, cw, h, finger=finger, source="fallback")
        if crop is not None:
            crops.append(crop)
    return crops


def crops_from_roi_overrides(rgb: np.ndarray, items: list[dict[str, object]]) -> list[NailCrop]:
    crops: list[NailCrop] = []
    for item in items:
        finger_raw = item.get("finger")
        finger: FingerLabel = "hand"
        if isinstance(finger_raw, str) and finger_raw in ("index", "middle", "ring", "hand"):
            finger = finger_raw  # type: ignore[assignment]
        try:
            x = float(item["x"])
            y = float(item["y"])
            w = float(item["w"])
            h = float(item["h"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PredictionServiceError(
                "Cada ROI requiere x, y, w, h numéricos.",
                422,
                code="invalid_rois_values",
            ) from exc
        crop = crop_from_normalized_roi(rgb, finger=finger, x=x, y=y, w=w, h=h)
        if crop is not None:
            crops.append(crop)
    return crops


class PassthroughNailDetector:
    """Un solo crop = imagen completa (tests / compatibilidad legacy)."""

    def detect(self, rgb_uint8: np.ndarray) -> list[NailCrop]:
        h, w = int(rgb_uint8.shape[0]), int(rgb_uint8.shape[1])
        return [NailCrop(finger="hand", rgb=rgb_uint8.copy(), bbox=(0, 0, w, h), source="fallback")]


class FixedNailDetector:
    """Detector inyectable para tests: devuelve N franjas verticales."""

    def __init__(self, count: int = 3) -> None:
        self._count = max(1, count)

    def detect(self, rgb_uint8: np.ndarray) -> list[NailCrop]:
        if self._count == 1:
            return PassthroughNailDetector().detect(rgb_uint8)
        h, w = int(rgb_uint8.shape[0]), int(rgb_uint8.shape[1])
        third = max(8, w // 3)
        labels: tuple[FingerLabel, ...] = ("index", "middle", "ring")
        crops: list[NailCrop] = []
        for i in range(min(self._count, 3)):
            x = i * third
            cw = third if i < 2 else w - x
            crop = _crop_axis_aligned(rgb_uint8, x, 0, cw, h, finger=labels[i])
            if crop is not None:
                crops.append(crop)
        return crops


class EmptyNailDetector:
    """Siempre devuelve lista vacía (tests de rechazo)."""

    def detect(self, rgb_uint8: np.ndarray) -> list[NailCrop]:
        _ = rgb_uint8
        return []


class MediaPipeNailDetector:
    """Detección de índice/medio/anular vía MediaPipe Hand Landmarker + recorte."""

    def __init__(
        self,
        *,
        min_confidence: float | None = None,
        crop_scale: float | None = None,
        fallback_mode: FallbackMode | None = None,
        roi_overrides: list[dict[str, object]] | None = None,
    ) -> None:
        self._min_confidence = (
            float(min_confidence)
            if min_confidence is not None
            else float(settings.predict_nail_detect_min_confidence)
        )
        self._crop_scale = (
            float(crop_scale) if crop_scale is not None else float(settings.predict_nail_crop_scale)
        )
        self._fallback_mode: FallbackMode = (
            fallback_mode if fallback_mode is not None else settings.predict_nail_fallback_mode
        )
        self._roi_overrides = roi_overrides

    def detect(self, rgb_uint8: np.ndarray) -> list[NailCrop]:
        if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
            return fallback_crops(rgb_uint8, self._fallback_mode)

        if self._roi_overrides is not None:
            return crops_from_roi_overrides(rgb_uint8, self._roi_overrides)

        try:
            import mediapipe as mp

            landmarker = _get_hand_landmarker(self._min_confidence)
        except (ImportError, FileNotFoundError, OSError) as exc:
            logger.warning(
                "Hand Landmarker no disponible (%s); usando fallback %s",
                exc,
                self._fallback_mode,
            )
            return fallback_crops(rgb_uint8, self._fallback_mode)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_uint8))
        from mediapipe.tasks.python.vision.core import image_processing_options as mp_ipo

        result = landmarker.detect(
            mp_image,
            image_processing_options=mp_ipo.ImageProcessingOptions(),
        )

        if not result.hand_landmarks:
            logger.info("MediaPipe: sin mano detectada; fallback=%s", self._fallback_mode)
            return fallback_crops(rgb_uint8, self._fallback_mode)

        landmarks = result.hand_landmarks[0]
        crops = landmarks_to_crops(
            rgb_uint8,
            landmarks,
            min_confidence=self._min_confidence,
            crop_scale=self._crop_scale,
        )

        if crops:
            return crops

        logger.info("MediaPipe: landmarks insuficientes; fallback=%s", self._fallback_mode)
        return fallback_crops(rgb_uint8, self._fallback_mode)


def build_nail_detector(*, roi_overrides: list[dict[str, object]] | None = None) -> NailDetector:
    """Factory con ROI manual opcional."""
    return MediaPipeNailDetector(roi_overrides=roi_overrides)
