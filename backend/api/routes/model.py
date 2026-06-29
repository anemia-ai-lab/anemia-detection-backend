from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.api.deps import ModelEvaluationServiceDep
from backend.schemas.model_evaluation import ModelEvaluationOut

router = APIRouter(tags=["model"])

_MODEL_EVALUATION_CACHE_MAX_AGE_S = 3600


@router.get(
    "/evaluation",
    response_model=ModelEvaluationOut,
    summary="Métricas offline del modelo",
    description="Métricas estáticas de evaluación alineadas con model_version.",
)
def model_evaluation(svc: ModelEvaluationServiceDep) -> JSONResponse:
    payload = svc.get_evaluation()
    return JSONResponse(
        content=payload.model_dump(mode="json"),
        headers={"Cache-Control": f"public, max-age={_MODEL_EVALUATION_CACHE_MAX_AGE_S}"},
    )
