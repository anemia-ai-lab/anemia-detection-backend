from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import ValidationError

from backend.api.deps import PredictContextDep, PredictionServiceDep
from backend.schemas.errors import ErrorResponse
from backend.schemas.prediction import (
    PredictionCreateBody,
    PredictionHistoryItem,
    PredictionImageSignedUrlOut,
    PredictionResponse,
)
from backend.services.exceptions import ClientHttpError, PredictionServiceError

router = APIRouter(tags=["predictions"])

_PREDICT_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {
        "model": ErrorResponse,
        "description": (
            "Imagen ausente/vacía, no decodificable, sin uña detectada (heurística), "
            "imagen demasiado pequeña u otros errores de imagen."
        ),
    },
    413: {
        "model": ErrorResponse,
        "description": "Fichero de imagen demasiado grande (límite configurado).",
    },
    415: {
        "model": ErrorResponse,
        "description": "Tipo MIME de imagen no soportado (solo JPEG, PNG, WebP).",
    },
    422: {
        "model": ErrorResponse,
        "description": "Validación de formulario (p. ej. birth_date inválida).",
    },
    401: {"model": ErrorResponse, "description": "Missing or invalid bearer token."},
    403: {"model": ErrorResponse, "description": "Ruta de imagen no permitida."},
    404: {"model": ErrorResponse, "description": "Predicción o imagen no encontrada."},
    503: {
        "model": ErrorResponse,
        "description": "Modelo Keras no cargado (INFERENCE_MODEL_PATH).",
    },
    502: {
        "model": ErrorResponse,
        "description": "Supabase/PostgREST error on read or write.",
    },
}


def _require_image_file(image: UploadFile | None) -> UploadFile:
    if image is None or not (image.filename and str(image.filename).strip()):
        raise PredictionServiceError(
            "image is required for prediction",
            400,
            code="image_required",
        )
    return image


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses=_PREDICT_RESPONSES,
    summary="Predicción de riesgo (imagen obligatoria)",
    description=(
        "Ejecuta inferencia **asistiva** sobre una imagen de uña (CNN MobileNetV2 + calibración por temperatura "
        "y umbral operacional configurados). Devuelve probabilidades, decisión binaria, nivel de riesgo y metadatos "
        "persistidos en Supabase.\n\n"
        "**Alcance:** estimación de riesgo para investigación o triaje informativo; **no** es diagnóstico clínico, "
        "no sustituye criterio médico ni analítica de laboratorio.\n\n"
        "Multipart: campo ``image`` obligatorio (JPEG, PNG o WebP); ``birth_date`` y ``notes`` opcionales."
    ),
)
async def predict(
    ctx: PredictContextDep,
    svc: PredictionServiceDep,
    image: Annotated[UploadFile | None, File(description="JPEG, PNG o WebP; máx. 5 MB.")] = None,
    birth_date: Annotated[str | None, Form()] = None,
    notes: Annotated[str | None, Form()] = None,
) -> PredictionResponse:
    upload = _require_image_file(image)
    user, access_token = ctx
    raw = await upload.read()
    if not raw:
        raise PredictionServiceError(
            "image is required for prediction",
            400,
            code="image_required",
        )
    fields: dict = {"notes": notes}
    if birth_date not in (None, ""):
        fields["birth_date"] = birth_date
    try:
        bd = PredictionCreateBody.model_validate(fields)
    except ValidationError as exc:
        raise ClientHttpError(
            "Validation error",
            422,
            code="validation_error",
        ) from exc
    return svc.run_predict(
        user,
        access_token,
        bd,
        raw,
        upload.content_type,
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
