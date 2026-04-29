"""
Inferencia TensorFlow Lite offline (G8) alineada con el backend.

- Preprocesado: ``ml.preprocessing.pipeline`` (mismo que Keras en API).
- Calibración / umbral: mismas funciones puras que ``POST /predict``.
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from backend.core.risk_mapping import RiskLevel, anemia_risk_label, risk_from_probability
from backend.inference.probability_calibration import (
    apply_temperature_calibration,
    binary_prediction_from_threshold,
)
from ml.preprocessing.pipeline import PreprocessingConfig, preprocess_image_bytes

logger = logging.getLogger(__name__)


class TFLiteMetadataError(ValueError):
    """Metadatos del export TFLite inválidos o incompletos."""


@dataclass(frozen=True)
class TFLiteExportMetadata:
    """Subconjunto validado del JSON generado por ``export_tflite.py``."""

    model_version: str
    temperature: float
    operational_threshold: float
    raw_output_is_sigmoid_probability: bool
    temperature_scaling_applied_inside_graph: bool

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> TFLiteExportMetadata:
        try:
            raw_sig = bool(data["raw_output_is_sigmoid_probability"])
            temp_inside = bool(data["temperature_scaling_applied_inside_graph"])
            mv = str(data["model_version"])
            t = float(data["temperature"])
            th = float(data["operational_threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TFLiteMetadataError("JSON de metadatos incompleto o tipos inválidos") from exc
        if not raw_sig:
            raise TFLiteMetadataError("se esperaba raw_output_is_sigmoid_probability=true")
        if temp_inside:
            raise TFLiteMetadataError("temperature_scaling_applied_inside_graph debe ser false")
        if t <= 0:
            raise TFLiteMetadataError("temperature debe ser > 0")
        return cls(
            model_version=mv,
            temperature=t,
            operational_threshold=th,
            raw_output_is_sigmoid_probability=raw_sig,
            temperature_scaling_applied_inside_graph=temp_inside,
        )


@dataclass(frozen=True)
class TFLiteInferenceResult:
    raw_probability: float
    calibrated_probability: float
    prediction: Literal[0, 1]
    risk: RiskLevel
    risk_label: str
    threshold_used: float
    temperature: float
    model_version: str
    inference_mode: Literal["tflite_offline"] = "tflite_offline"
    created_at: str = ""
    image_sha256: str = ""
    preprocessing: dict[str, Any] | None = None

    def to_sync_payload(self) -> dict[str, Any]:
        """
        Carga útil alineada con ``PredictionResponse`` (campos omitidos si no existen en offline).

        Pensado para futura ingesta batch en el backend (p. ej. sincronización de campo).
        """
        return {
            "raw_probability": self.raw_probability,
            "calibrated_probability": self.calibrated_probability,
            "score": self.calibrated_probability,
            "threshold_used": self.threshold_used,
            "prediction": self.prediction,
            "risk": self.risk,
            "risk_label": self.risk_label,
            "message": self.risk_label,
            "model_version": self.model_version,
            "inference_mode": self.inference_mode,
            "created_at": self.created_at,
            "image_sha256": self.image_sha256,
            "preprocessing": self.preprocessing or {},
        }

    def to_json_dict(self) -> dict[str, Any]:
        """Serialización JSON-friendly (sin tipos numpy)."""
        d = asdict(self)
        d["preprocessing"] = dict(self.preprocessing or {})
        return d


class TFLiteInferenceEngine:
    """Carga perezosa del intérprete TFLite + metadatos de calibración."""

    def __init__(
        self,
        tflite_path: Path,
        metadata_path: Path,
        *,
        preprocess_cfg: PreprocessingConfig | None = None,
    ) -> None:
        self._tflite_path = Path(tflite_path).resolve()
        self._metadata_path = Path(metadata_path).resolve()
        self._preprocess_cfg = preprocess_cfg or PreprocessingConfig()
        self._meta: TFLiteExportMetadata | None = None
        self._interpreter: Any = None

    def _ensure_loaded(self) -> None:
        if self._interpreter is not None:
            return
        if not self._tflite_path.is_file():
            msg = f"No existe el fichero TFLite: {self._tflite_path}"
            raise FileNotFoundError(msg)
        if not self._metadata_path.is_file():
            msg = f"No existe el JSON de metadatos: {self._metadata_path}"
            raise FileNotFoundError(msg)
        raw = self._metadata_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise TFLiteMetadataError("metadatos: se esperaba un objeto JSON")
        self._meta = TFLiteExportMetadata.from_json_dict(data)

        import tensorflow as tf

        self._interpreter = tf.lite.Interpreter(model_path=str(self._tflite_path))
        self._interpreter.allocate_tensors()
        logger.info(
            "tflite_engine_loaded path=%s metadata=%s model_version=%s",
            self._tflite_path,
            self._metadata_path,
            self._meta.model_version,
        )

    @property
    def metadata(self) -> TFLiteExportMetadata:
        self._ensure_loaded()
        assert self._meta is not None
        return self._meta

    def predict(self, image_bytes: bytes) -> TFLiteInferenceResult:
        return self.predict_batch([image_bytes])[0]

    def predict_batch(self, images: Sequence[bytes]) -> list[TFLiteInferenceResult]:
        self._ensure_loaded()
        assert self._meta is not None and self._interpreter is not None

        inp = self._interpreter.get_input_details()[0]
        out = self._interpreter.get_output_details()[0]

        results: list[TFLiteInferenceResult] = []
        for raw in images:
            pre = preprocess_image_bytes(raw, cfg=self._preprocess_cfg)
            batch = pre.model_input_tensor.astype(np.float32)
            if batch.shape != tuple(inp["shape"]):
                try:
                    self._interpreter.resize_tensor_input(inp["index"], list(batch.shape))
                    self._interpreter.allocate_tensors()
                    inp = self._interpreter.get_input_details()[0]
                    out = self._interpreter.get_output_details()[0]
                except ValueError as exc:
                    msg = f"forma de entrada incompatible: {batch.shape} vs {inp['shape']}"
                    raise ValueError(msg) from exc

            self._interpreter.set_tensor(inp["index"], batch)
            self._interpreter.invoke()
            y = np.copy(self._interpreter.get_tensor(out["index"]))
            raw_p = float(np.asarray(y).squeeze())
            raw_p = max(0.0, min(1.0, raw_p))

            cal = apply_temperature_calibration(raw_p, self._meta.temperature)
            th = float(self._meta.operational_threshold)
            pred = int(binary_prediction_from_threshold(cal, th))
            risk = risk_from_probability(cal, th)
            label = anemia_risk_label(risk)
            ts = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            h = pre.provenance.get("raw_bytes_sha256") or ""
            results.append(
                TFLiteInferenceResult(
                    raw_probability=raw_p,
                    calibrated_probability=cal,
                    prediction=pred,
                    risk=risk,
                    risk_label=label,
                    threshold_used=th,
                    temperature=float(self._meta.temperature),
                    model_version=self._meta.model_version,
                    created_at=ts,
                    image_sha256=h,
                    preprocessing=dict(pre.provenance),
                ),
            )
        return results


@functools.lru_cache(maxsize=8)
def get_tflite_engine(tflite_path: str, metadata_path: str) -> TFLiteInferenceEngine:
    """Fábrica con caché por rutas; carga perezosa en ``TFLiteInferenceEngine``."""
    return TFLiteInferenceEngine(Path(tflite_path), Path(metadata_path))
