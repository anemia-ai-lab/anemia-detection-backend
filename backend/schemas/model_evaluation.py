from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

# Última evaluación documentada: calibración post-hoc (test) alineada con la tesis.
_DEFAULT_EVAL_AT = datetime(2026, 4, 20, 4, 50, 56, 897574, tzinfo=UTC)


class ModelEvalMetrics(BaseModel):
    """
    Métricas offline del mejor pipeline (entrenamiento + calibración en test).

    Origen: ``experiment_20260420T043804Z`` (train/val/test) y
    ``calibration_20260420T045056Z`` (temperature scaling + métricas en probabilidades calibradas).
    La versión de despliegue del API va en ``model_version`` (p. ej. ``v1.0``).
    """

    auc: float = Field(
        default=0.795092,
        ge=0.0,
        le=1.0,
        description="AUC-ROC en test (Keras; invariante ante escalado monótono del score).",
    )
    precision_operational: float = Field(
        default=0.454545,
        ge=0.0,
        le=1.0,
        description="Precisión en test al umbral operacional (Youden) sobre **probabilidad calibrada**.",
    )
    recall_operational: float = Field(
        default=0.740741,
        ge=0.0,
        le=1.0,
        description="Recall (sensibilidad) en test al umbral operacional sobre probabilidad calibrada.",
    )
    accuracy_operational: float = Field(
        default=0.793333,
        ge=0.0,
        le=1.0,
        description="Exactitud en test al umbral operacional sobre probabilidad calibrada.",
    )
    operational_threshold: float = Field(
        default=0.1680544387290045,
        ge=0.0,
        le=1.0,
        description="Umbral τ de Youden (ROC) aplicado sobre la probabilidad **calibrada** en test.",
    )
    temperature: float = Field(
        default=0.7510018331928743,
        gt=0.0,
        description="Parámetro T de *temperature scaling* ajustado en validación (inferencia: ``sigmoid(logit(p)/T)``).",
    )
    brier_score: float = Field(
        default=0.11766287029947034,
        ge=0.0,
        description="Brier score en test con probabilidades calibradas.",
    )
    expected_calibration_error: float = Field(
        default=0.060344067420365466,
        ge=0.0,
        le=1.0,
        description="ECE (error esperado de calibración) en test, probabilidades calibradas (15 bins).",
    )
    oversampling_used: bool = Field(
        default=True,
        description="Oversampling de positivos en el train del ``fit`` (~1:1 en subconjunto interno).",
    )
    class_weight_used: bool = Field(
        default=False,
        description="Si ``class_weight`` se aplicó en ``model.fit`` (este modelo: no, ``--no-class-weight``).",
    )
    fine_tuning_used: bool = Field(
        default=True,
        description="Si hubo segunda fase de fine-tuning parcial del backbone MobileNetV2.",
    )
    evaluated_at: datetime = Field(
        default=_DEFAULT_EVAL_AT,
        description="Marca temporal de la evaluación/calibración documentada (UTC).",
    )
    dataset_version: str = Field(
        default="experiment_20260420T043804Z; calibration_20260420T045056Z",
        description="Trazabilidad de los artefactos JSON de entrenamiento y calibración.",
    )


class ModelEvaluationOut(ModelEvalMetrics):
    """
    Cuerpo de ``GET /model/evaluation``: métricas offline + ``model_version`` coherente con ``POST /predict``.

    Sirve como **cita reproducible** en memoria de tesis o artículo (tabla de rendimiento en test); no ejecuta
    el modelo ni produce juicio sobre un individuo concreto.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_version": "v1.0",
                "auc": 0.795092,
                "precision_operational": 0.454545,
                "recall_operational": 0.740741,
                "accuracy_operational": 0.793333,
                "operational_threshold": 0.1680544387290045,
                "temperature": 0.7510018331928743,
                "brier_score": 0.11766287029947034,
                "expected_calibration_error": 0.060344067420365466,
                "oversampling_used": True,
                "class_weight_used": False,
                "fine_tuning_used": True,
                "evaluated_at": "2026-04-20T04:50:56.897574Z",
                "dataset_version": "experiment_20260420T043804Z; calibration_20260420T045056Z",
            }
        },
    )

    model_version: str = Field(examples=["v1.0"])
