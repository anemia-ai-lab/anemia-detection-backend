from fastapi import APIRouter

from backend.api.deps import ModelEvaluationServiceDep
from backend.schemas.model_evaluation import ModelEvaluationOut

router = APIRouter(tags=["model"])


@router.get(
    "/evaluation",
    response_model=ModelEvaluationOut,
    summary="Métricas offline del modelo (configuración)",
    description=(
        "Devuelve métricas de **evaluación documentada** en test (AUC, precisión/recall/exactitud al umbral "
        "operacional, temperatura de calibración, ECE, Brier, flags de entrenamiento, etc.), alineadas con la "
        "versión ``model_version`` usada en ``POST /predict``. Los valores provienen de configuración estática, "
        "no de inferencia en tiempo real. **Referencia metodológica; no es predicción ni diagnóstico sobre pacientes.**"
    ),
)
def model_evaluation(svc: ModelEvaluationServiceDep) -> ModelEvaluationOut:
    return svc.get_evaluation()
