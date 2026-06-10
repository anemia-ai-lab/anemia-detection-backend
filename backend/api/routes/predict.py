from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Response, UploadFile

from backend.api.deps import PredictContextDep, PredictionServiceDep
from backend.schemas.errors import ErrorResponse
from backend.schemas.prediction import (
    PredictionDetailOut,
    PredictionImageSignedUrlOut,
    PredictionImageUploadOut,
    PredictionListResponse,
    PredictionResponse,
    PredictionSyncMetadataRequest,
    PredictionSyncMetadataResponse,
)

router = APIRouter(tags=["predictions"])

# POST /predict: lectura multipart en el servicio; la ruta solo enlaza Starlette/FastAPI.

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
    409: {
        "model": ErrorResponse,
        "description": "image_sha256 no coincide con la imagen enviada.",
    },
    503: {
        "model": ErrorResponse,
        "description": "Modelo Keras no cargado (INFERENCE_MODEL_PATH).",
    },
    502: {
        "model": ErrorResponse,
        "description": "Supabase/PostgREST error on read or write.",
    },
}


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
    user, access_token = ctx
    return await svc.run_predict_from_upload(
        user,
        access_token,
        image=image,
        birth_date=birth_date,
        notes=notes,
    )


@router.post(
    "/predictions/sync/metadata",
    response_model=PredictionSyncMetadataResponse,
    responses=_PREDICT_RESPONSES,
    summary="Sync offline — paso 1: metadatos en batch",
    description=(
        "Ingesta idempotente de predicciones hechas offline (TFLite). "
        "Cada item requiere ``client_id`` único en el dispositivo. "
        "Las imágenes se suben en el paso 2 (``POST /predictions/{id}/image``)."
    ),
)
def sync_predictions_metadata(
    body: PredictionSyncMetadataRequest,
    ctx: PredictContextDep,
    svc: PredictionServiceDep,
) -> PredictionSyncMetadataResponse:
    user, access_token = ctx
    return svc.sync_metadata_batch(user, access_token, body)


@router.get(
    "/predictions",
    response_model=PredictionListResponse,
    responses=_PREDICT_RESPONSES,
    summary="List my predictions (paginated)",
)
def list_predictions(
    ctx: PredictContextDep,
    svc: PredictionServiceDep,
    limit: Annotated[int | None, Query(ge=1, le=100)] = None,
    cursor: Annotated[str | None, Query()] = None,
) -> PredictionListResponse:
    """History for the JWT subject (RLS limits rows to the current user)."""
    _user, access_token = ctx
    return svc.list_predictions_paginated(access_token, limit=limit, cursor=cursor)


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionDetailOut,
    responses=_PREDICT_RESPONSES,
    summary="Detalle de una predicción propia",
)
def get_prediction(
    prediction_id: str,
    ctx: PredictContextDep,
    svc: PredictionServiceDep,
) -> PredictionDetailOut:
    user, access_token = ctx
    return svc.get_prediction_detail(user, access_token, prediction_id)


@router.delete(
    "/predictions/{prediction_id}",
    status_code=204,
    responses=_PREDICT_RESPONSES,
    summary="Borrar una predicción propia",
)
def delete_prediction(
    prediction_id: str,
    ctx: PredictContextDep,
    svc: PredictionServiceDep,
) -> Response:
    user, access_token = ctx
    svc.delete_prediction(user, access_token, prediction_id)
    return Response(status_code=204)


@router.post(
    "/predictions/{prediction_id}/image",
    response_model=PredictionImageUploadOut,
    responses=_PREDICT_RESPONSES,
    summary="Sync offline — paso 2: subir imagen",
)
async def upload_prediction_image(
    prediction_id: str,
    ctx: PredictContextDep,
    svc: PredictionServiceDep,
    image: Annotated[UploadFile | None, File(description="JPEG, PNG o WebP; máx. 5 MB.")] = None,
    image_sha256: Annotated[str | None, Form()] = None,
) -> PredictionImageUploadOut:
    user, access_token = ctx
    return await svc.upload_prediction_image(
        user,
        access_token,
        prediction_id,
        image=image,
        image_sha256=image_sha256,
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
