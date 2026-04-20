from typing import Annotated

from fastapi import APIRouter, Body, File, Form, UploadFile

from backend.api.deps import PredictContextDep, PredictionServiceDep
from backend.schemas.errors import ErrorResponse
from backend.schemas.prediction import (
    PredictionCreateBody,
    PredictionHistoryItem,
    PredictionImageSignedUrlOut,
    PredictionResponse,
)

router = APIRouter(tags=["predictions"])

_PREDICT_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ErrorResponse, "description": "Imagen inválida o demasiado grande."},
    401: {"model": ErrorResponse, "description": "Missing or invalid bearer token."},
    403: {"model": ErrorResponse, "description": "Ruta de imagen no permitida."},
    404: {"model": ErrorResponse, "description": "Predicción o imagen no encontrada."},
    502: {
        "model": ErrorResponse,
        "description": "Supabase/PostgREST error on read or write.",
    },
}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses=_PREDICT_RESPONSES,
    summary="Run mock prediction (persisted)",
)
def predict(
    ctx: PredictContextDep,
    svc: PredictionServiceDep,
    body: PredictionCreateBody = Body(default_factory=PredictionCreateBody),
) -> PredictionResponse:
    user, access_token = ctx
    return svc.run_predict(user, access_token, body)


@router.post(
    "/predict/with-image",
    response_model=PredictionResponse,
    responses=_PREDICT_RESPONSES,
    summary="Predicción con imagen (Storage, JWT del usuario)",
)
async def predict_with_image(
    ctx: PredictContextDep,
    svc: PredictionServiceDep,
    image: Annotated[UploadFile, File(description="JPEG, PNG o WebP; máx. 5 MB.")],
    birth_date: Annotated[str | None, Form()] = None,
    notes: Annotated[str | None, Form()] = None,
) -> PredictionResponse:
    user, access_token = ctx
    raw = await image.read()
    fields: dict = {"notes": notes}
    if birth_date not in (None, ""):
        fields["birth_date"] = birth_date
    bd = PredictionCreateBody.model_validate(fields)
    return svc.run_predict_with_image(
        user,
        access_token,
        bd,
        raw,
        image.content_type,
    )


@router.get(
    "/predictions/{prediction_id}/image-signed-url",
    response_model=PredictionImageSignedUrlOut,
    responses=_PREDICT_RESPONSES,
    summary="URL firmada temporal para la imagen de una predicción propia",
)
def prediction_image_signed_url(
    prediction_id: str,
    ctx: PredictContextDep,
    svc: PredictionServiceDep,
) -> PredictionImageSignedUrlOut:
    user, access_token = ctx
    return svc.signed_image_url_for_prediction(user, access_token, prediction_id)


@router.get(
    "/predictions",
    response_model=list[PredictionHistoryItem],
    responses=_PREDICT_RESPONSES,
    summary="List my predictions",
)
def list_predictions(
    ctx: PredictContextDep,
    svc: PredictionServiceDep,
) -> list[PredictionHistoryItem]:
    """History for the JWT subject (RLS limits rows to the current user)."""
    _user, access_token = ctx
    return svc.list_predictions(access_token)
