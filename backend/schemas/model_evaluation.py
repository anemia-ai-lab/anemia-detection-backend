from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

# Última evaluación documentada: ensemble unique-hash + Platt (test Ghana).
_DEFAULT_EVAL_AT = datetime(2026, 9, 14, 14, 43, 34, 746424, tzinfo=UTC)


class ModelEvalMetrics(BaseModel):
    """
    Métricas offline del pipeline pediátrico v2 (ensemble 3 semillas + calibración en test).

    Origen: ``calibration_ensemble_ghana_v2`` (Platt en unique-hash; T queda como rollback).
    La versión de despliegue del API va en ``model_version`` (p. ej. ``v2.0``).
    """

    auc: float = Field(
        default=0.639933,
        ge=0.0,
        le=1.0,
        description="AUC-ROC en test (Keras; invariante ante escalado monótono del score).",
    )
    precision_operational: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Precisión en test al umbral operacional (Youden) sobre **probabilidad calibrada**.",
    )
    recall_operational: float = Field(
        default=0.62201,
        ge=0.0,
        le=1.0,
        description="Recall (sensibilidad) en test al umbral operacional sobre probabilidad calibrada.",
    )
    accuracy_operational: float = Field(
        default=0.621827,
        ge=0.0,
        le=1.0,
        description="Exactitud en test al umbral operacional sobre probabilidad calibrada.",
    )
    operational_threshold: float = Field(
        default=0.5780355600619943,
        ge=0.0,
        le=1.0,
        description="Umbral τ de Youden (ROC) aplicado sobre la probabilidad **calibrada** en test.",
    )
    temperature: float = Field(
        default=0.9443417710165931,
        gt=0.0,
        description="Parámetro T de *temperature scaling* (rollback; inferencia default = Platt).",
    )
    brier_score: float = Field(
        default=0.236664,
        ge=0.0,
        description="Brier score en test con probabilidades calibradas.",
    )
    expected_calibration_error: float = Field(
        default=0.065715,
        ge=0.0,
        le=1.0,
        description="ECE (error esperado de calibración) en test, probabilidades calibradas (15 bins).",
    )
    oversampling_used: bool = Field(
        default=False,
        description="Oversampling de positivos en el train del ``fit`` (~1:1 en subconjunto interno).",
    )
    class_weight_used: bool = Field(
        default=False,
        description="Si ``class_weight`` se aplicó en ``model.fit``.",
    )
    fine_tuning_used: bool = Field(
        default=False,
        description="Si hubo segunda fase de fine-tuning parcial del backbone MobileNetV2.",
    )
    evaluated_at: datetime = Field(
        default=_DEFAULT_EVAL_AT,
        description="Marca temporal de la evaluación/calibración documentada (UTC).",
    )
    dataset_version: str = Field(
        default="calibration_ensemble_ghana_v2",
        description="Trazabilidad del artefacto JSON de calibración final.",
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
                "model_version": "v2.0",
                "auc": 0.639933,
                "precision_operational": 0.65,
                "recall_operational": 0.62201,
                "accuracy_operational": 0.621827,
                "operational_threshold": 0.5780355600619943,
                "temperature": 0.9443417710165931,
                "brier_score": 0.236664,
                "expected_calibration_error": 0.065715,
                "oversampling_used": False,
                "class_weight_used": False,
                "fine_tuning_used": False,
                "evaluated_at": "2026-09-14T14:43:34.746424Z",
                "dataset_version": "calibration_ensemble_ghana_v2",
            }
        },
    )

    model_version: str = Field(examples=["v2.0"])
