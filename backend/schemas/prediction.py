"""Prediction API contracts."""

from datetime import date, datetime
from typing import Any, Literal, Optional, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.core import patient_age
from backend.core.risk_mapping import RiskLevel

SYNC_METADATA_BATCH_MAX = 50


class PredictionCreateBody(BaseModel):
    """Campos opcionales del formulario multipart de ``POST /predict`` (además de la imagen)."""

    birth_date: Optional[date] = None
    notes: Optional[str] = Field(default=None, max_length=4000)

    @field_validator("birth_date")
    @classmethod
    def birth_date_plausible_utc(cls, v: date | None) -> date | None:
        if v is None:
            return v
        ref = patient_age.utc_today()
        if v > ref:
            raise ValueError("La fecha de nacimiento no puede ser futura (UTC).")
        if v < patient_age.min_plausible_birth_date(ref):
            raise ValueError("La fecha de nacimiento no es plausible.")
        return v


class PredictionListItem(BaseModel):
    """Resumen de una fila en ``GET /predictions`` (sin imagen ni preprocessing)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "00000000-0000-0000-0000-000000000000",
                "risk": "low",
                "score": 0.42,
                "model_version": "v2.0",
                "birth_date": "2016-01-15",
                "age_months": 111,
                "age_display": "9 años 3 meses",
                "notes": None,
                "effective_created_at": "2026-04-01T12:00:00Z",
                "inference_mode": "backend",
                "has_image": True,
            }
        },
    )

    id: str
    risk: RiskLevel
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilidad calibrada de clase positiva [0, 1].",
    )
    model_version: str
    birth_date: Optional[date] = None
    age_months: Optional[int] = None
    age_display: Optional[str] = Field(
        default=None,
        description="Edad legible desde age_months (es), p. ej. 9 años 3 meses.",
    )
    notes: Optional[str] = None
    effective_created_at: datetime = Field(
        description="COALESCE(client_created_at, created_at) para orden cronológico.",
    )
    inference_mode: Literal["backend", "tflite_offline"] = Field(
        default="backend",
        description="Dónde se ejecutó la inferencia.",
    )
    has_image: bool = Field(description="True si la imagen ya está en Storage.")


class PredictionListResponse(BaseModel):
    """Lista paginada de predicciones del usuario autenticado."""

    items: list[PredictionListItem]
    next_cursor: Optional[str] = Field(
        default=None,
        description="Cursor opaco para la página siguiente; null si no hay más.",
    )


# Alias retrocompatible en OpenAPI interno / tests legacy.
PredictionHistoryItem = PredictionListItem


class PredictionDetailOut(BaseModel):
    """Detalle de ``GET /predictions/{id}`` con metadatos completos e imagen firmada."""

    id: str
    risk: RiskLevel
    score: float = Field(ge=0.0, le=1.0)
    raw_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    calibrated_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    threshold_used: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    prediction: Optional[Literal[0, 1]] = None
    risk_label: str
    model_version: str
    birth_date: Optional[date] = None
    age_months: Optional[int] = None
    age_display: Optional[str] = None
    notes: Optional[str] = None
    client_id: Optional[str] = None
    inference_mode: Literal["backend", "tflite_offline"] = "backend"
    client_created_at: Optional[datetime] = None
    effective_created_at: datetime
    created_at: datetime
    synced_at: Optional[datetime] = None
    image_sha256: Optional[str] = None
    preprocessing: Optional[dict[str, Any]] = None
    has_image: bool
    image_signed_url: Optional[str] = Field(
        default=None,
        description="URL firmada (~1 h) si ``has_image``.",
    )


class PredictionSyncMetadataItem(BaseModel):
    """Metadatos de una predicción offline (paso 1 del sync)."""

    client_id: str = Field(description="UUID v4 generado en el dispositivo.")
    risk: RiskLevel
    score: float = Field(ge=0.0, le=1.0)
    raw_probability: float = Field(ge=0.0, le=1.0)
    calibrated_probability: float = Field(ge=0.0, le=1.0)
    threshold_used: float = Field(ge=0.0, le=1.0)
    prediction: Literal[0, 1]
    model_version: str
    inference_mode: Literal["tflite_offline"] = "tflite_offline"
    client_created_at: datetime
    birth_date: Optional[date] = None
    notes: Optional[str] = Field(default=None, max_length=4000)
    image_sha256: Optional[str] = Field(default=None, max_length=128)
    preprocessing: Optional[dict[str, Any]] = None

    @field_validator("birth_date")
    @classmethod
    def birth_date_plausible_utc(cls, v: date | None) -> date | None:
        if v is None:
            return v
        ref = patient_age.utc_today()
        if v > ref:
            raise ValueError("La fecha de nacimiento no puede ser futura (UTC).")
        if v < patient_age.min_plausible_birth_date(ref):
            raise ValueError("La fecha de nacimiento no es plausible.")
        return v

    @field_validator("preprocessing")
    @classmethod
    def preprocessing_is_object(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is None:
            return v
        if not isinstance(v, dict):
            raise ValueError("preprocessing debe ser un objeto JSON.")
        return v


class PredictionSyncMetadataRequest(BaseModel):
    items: list[PredictionSyncMetadataItem] = Field(min_length=1)

    @model_validator(mode="after")
    def batch_size_limit(self) -> Self:
        if len(self.items) > SYNC_METADATA_BATCH_MAX:
            raise ValueError(f"Máximo {SYNC_METADATA_BATCH_MAX} items por solicitud de sync.")
        return self


class PredictionSyncMetadataResult(BaseModel):
    client_id: str
    id: str
    image_pending: bool
    created: bool = Field(description="False si ya existía (reintento idempotente).")


class PredictionSyncMetadataResponse(BaseModel):
    results: list[PredictionSyncMetadataResult]


class PredictionImageUploadOut(BaseModel):
    """Respuesta de ``POST /predictions/{id}/image``."""

    id: str
    image_storage_path: str
    image_signed_url: str
    image_sha256: Optional[str] = None


class PredictionImageSignedUrlOut(BaseModel):
    """URL firmada temporal para leer la imagen (no se persiste en ``predictions``)."""

    signed_url: str = Field(description="Enlace firmado a Storage (~1 h).")


class PredictionResponse(BaseModel):
    """
    Éxito de ``POST /predict``: fila persistida, probabilidades y decisión binaria **asistida por modelo**.

    La API expone **predicción de riesgo** (salida del pipeline CNN + calibración configurada), no diagnóstico
    clínico ni recomendación terapéutica.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "00000000-0000-0000-0000-000000000000",
                "risk": "low",
                "score": 0.12,
                "raw_probability": 0.18,
                "calibrated_probability": 0.12,
                "threshold_used": 0.168,
                "prediction": 0,
                "risk_label": "Low anemia risk prediction",
                "message": "Low anemia risk prediction",
                "model_version": "v2.0",
                "birth_date": "2016-01-15",
                "age_months": 111,
                "age_display": "9 años 3 meses",
                "notes": None,
                "image_storage_path": "uuid/abc.jpg",
                "created_at": "2026-04-01T12:00:00Z",
                "inference_mode": "backend",
            }
        },
    )

    id: str = Field(description="Row id in `public.predictions`.")
    risk: RiskLevel
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Probabilidad **calibrada** de clase positiva persistida en BD (misma escala que "
            "``calibrated_probability``; alineada con ``threshold_used`` de la tesis)."
        ),
    )
    raw_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Salida sigmoide del modelo CNN sin post-proceso (antes de *temperature scaling*).",
    )
    calibrated_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilidad tras ``sigmoid(logit(raw) / T)`` en inferencia (T desde configuración).",
    )
    threshold_used: float = Field(
        ge=0.0,
        le=1.0,
        description="Umbral alto (τ_alto / high_lower) para decisión binaria y riesgo ``high``.",
    )
    risk_tier_low_upper: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Umbral bajo: probabilidad calibrada ≤ valor → riesgo ``low``.",
    )
    risk_tier_high_lower: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Umbral alto: probabilidad calibrada ≥ valor → riesgo ``high``.",
    )
    prediction: Literal[0, 1] = Field(
        description="Decisión binaria: 1 si ``calibrated_probability >= threshold_used``, si no 0.",
    )
    risk_label: str = Field(
        description="Resumen legible alineado con ``risk`` y ``prediction`` (demos / OpenAPI).",
    )
    message: Optional[str] = Field(
        default=None,
        description=(
            "Mensaje humano opcional; si se incluye, coincide con ``risk_label`` y con ``risk``/``prediction``. "
            "**Predicción asistiva, no diagnóstico médico.**"
        ),
    )
    model_version: str
    birth_date: Optional[date] = None
    age_months: Optional[int] = None
    age_display: Optional[str] = Field(
        default=None,
        description="Edad legible desde age_months (es).",
    )
    notes: Optional[str] = None
    image_storage_path: Optional[str] = Field(
        default=None,
        description="Ruta del objeto en Storage.",
    )
    created_at: datetime
    inference_mode: Literal["backend", "tflite_offline"] = Field(
        default="backend",
        description="Dónde se ejecutó la inferencia (API backend u offline TFLite para sincronización).",
    )
