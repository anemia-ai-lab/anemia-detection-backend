"""Prediction API contracts."""

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.core import patient_age
from backend.core.risk_mapping import RiskLevel


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


class PredictionHistoryItem(BaseModel):
    """One row returned by ``GET /predictions`` (newest first)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "00000000-0000-0000-0000-000000000000",
                "risk": "low",
                "score": 0.42,
                "model_version": "v1.0",
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

    id: str
    risk: RiskLevel
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilidad estimada de la clase positiva (v1, [0, 1]).",
    )
    model_version: str
    birth_date: Optional[date] = None
    age_months: Optional[int] = None
    age_display: Optional[str] = Field(
        default=None,
        description="Edad legible desde age_months (es), p. ej. 9 años 3 meses.",
    )
    notes: Optional[str] = None
    image_storage_path: Optional[str] = Field(
        default=None,
        description="Ruta del objeto en el bucket de Storage (prefijo = user id).",
    )
    created_at: datetime
    inference_mode: Literal["backend", "tflite_offline"] = Field(
        default="backend",
        description="Dónde se ejecutó la inferencia (API backend u offline TFLite para sincronización).",
    )


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
                "model_version": "v1.0",
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
        description="Umbral operacional aplicado sobre ``calibrated_probability`` (ROC-Youden en evaluación).",
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
