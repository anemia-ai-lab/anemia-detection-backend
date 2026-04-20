from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

_DEFAULT_EVAL_AT = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


class ModelEvalMetrics(BaseModel):
    """Métricas offline; la versión del modelo va en ``settings.model_version``."""

    sensitivity: float = Field(default=0.82, ge=0.0, le=1.0, examples=[0.82])
    specificity: float = Field(default=0.79, ge=0.0, le=1.0, examples=[0.79])
    f1_score: float = Field(default=0.80, ge=0.0, le=1.0, examples=[0.8])
    auc_roc: float = Field(default=0.88, ge=0.0, le=1.0, examples=[0.88])
    inference_time_ms: float = Field(default=45.0, ge=0.0, examples=[45.0])
    model_size_mb: float = Field(default=12.5, ge=0.0, examples=[12.5])
    grad_cam_available: bool = Field(default=True, examples=[True])
    evaluated_at: datetime = Field(
        default=_DEFAULT_EVAL_AT,
        description="Momento en que se fijaron estas métricas (evaluación offline).",
    )
    dataset_version: str = Field(default="internal-v1", examples=["internal-v1"])


class ModelEvaluationOut(ModelEvalMetrics):
    """Respuesta de ``GET /model/evaluation`` (versión alineada con predicciones)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_version": "v1.0",
                "sensitivity": 0.82,
                "specificity": 0.79,
                "f1_score": 0.8,
                "auc_roc": 0.88,
                "inference_time_ms": 45.0,
                "model_size_mb": 12.5,
                "grad_cam_available": True,
                "evaluated_at": "2026-01-15T12:00:00Z",
                "dataset_version": "internal-v1",
            }
        },
    )

    model_version: str = Field(examples=["v1.0"])
