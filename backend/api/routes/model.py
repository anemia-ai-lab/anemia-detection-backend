from fastapi import APIRouter

from backend.api.deps import ModelEvaluationServiceDep
from backend.schemas.model_evaluation import ModelEvaluationOut

router = APIRouter(tags=["model"])


@router.get(
    "/evaluation",
    response_model=ModelEvaluationOut,
    summary="Métricas offline del modelo",
    description="Métricas estáticas de evaluación alineadas con model_version.",
)
def model_evaluation(svc: ModelEvaluationServiceDep) -> ModelEvaluationOut:
    return svc.get_evaluation()
