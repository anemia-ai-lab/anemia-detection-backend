"""Contrato de ``GET /health``."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthOut(BaseModel):
    """
    Estado operativo del servicio y de la ruta de inferencia.

    Pensado para orquestación (liveness/readiness simplificado), demos y anexos de tesis; **no** sustituye
    monitorización clínica ni auditoría de modelos en producción avanzada.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "model_loaded": True,
                "model_version": "v1.0",
                "calibration_enabled": True,
                "inference_model_path": "ml/artifacts/models/baseline_mobilenetv2.keras",
            }
        },
    )

    status: Literal["ok", "degraded"] = Field(
        description=(
            "``ok``: proceso sano y, si hay ruta de modelo, el fichero existe y el cargador Keras terminó bien "
            "(o bien no hay modelo configurado, intencionalmente). "
            "``degraded``: ruta configurada pero fichero ausente o fallo al cargar el ``.keras``."
        ),
    )
    model_loaded: bool = Field(
        description="``True`` si hay predictor Keras listo en este proceso (``POST /predict`` puede inferir).",
    )
    model_version: str = Field(
        description="Etiqueta de versión desplegada (misma que en predicciones persistidas y ``GET /model/evaluation``).",
    )
    calibration_enabled: bool = Field(
        description="``True`` si T≠1 en configuración (temperature scaling no trivial en inferencia).",
    )
    inference_model_path: str | None = Field(
        default=None,
        description="Valor literal de ``INFERENCE_MODEL_PATH`` del entorno (relativo al repo o absoluto); omitido si vacío.",
    )
