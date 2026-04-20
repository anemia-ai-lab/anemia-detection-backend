from backend.core.config import settings
from backend.schemas.model_evaluation import ModelEvaluationOut


class ModelEvaluationService:
    """Expone métricas de evaluación del modelo desde configuración (sin inferencia en vivo)."""

    def get_evaluation(self) -> ModelEvaluationOut:
        return ModelEvaluationOut(
            model_version=settings.model_version,
            **settings.model_eval.model_dump(),
        )
