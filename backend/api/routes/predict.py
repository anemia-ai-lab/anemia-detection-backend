from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Response, UploadFile

from backend.api.deps import PredictContextDep, PredictionServiceDep
from backend.schemas.errors import ErrorResponse
from backend.schemas.prediction import (
    PredictionDetailOut,
    PredictionImageUploadOut,
    PredictionListResponse,
    PredictionResponse,
    PredictionSyncMetadataRequest,
    PredictionSyncMetadataResponse,
)

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
    summary="Inferencia online",
    description="Multipart: ``image`` (JPEG/PNG/WebP); ``birth_date`` y ``notes`` opcionales.",
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


@router.get(
    "/predictions",
    response_model=PredictionListResponse,
    responses=_PREDICT_RESPONSES,
    summary="Historial paginado",
    description="Query: ``limit`` (1–100), ``cursor`` opaco de la página anterior.",
)
def list_predictions(
    ctx: PredictContextDep,
    svc: PredictionServiceDep,
    limit: Annotated[int | None, Query(ge=1, le=100)] = None,
    cursor: Annotated[str | None, Query()] = None,
) -> PredictionListResponse:
    _user, access_token = ctx
    return svc.list_predictions_paginated(access_token, limit=limit, cursor=cursor)


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionDetailOut,
    responses=_PREDICT_RESPONSES,
    summary="Detalle de predicción",
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
    summary="Borrar predicción",
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
    "/predictions/sync/metadata",
    response_model=PredictionSyncMetadataResponse,
    responses=_PREDICT_RESPONSES,
    tags=["offline-sync"],
    summary="Sync offline — paso 1: metadatos",
    description="Idempotente por ``client_id``; máx. 50 items. Imagen en paso 2.",
)
def sync_predictions_metadata(
    body: PredictionSyncMetadataRequest,
    ctx: PredictContextDep,
    svc: PredictionServiceDep,
) -> PredictionSyncMetadataResponse:
    user, access_token = ctx
    return svc.sync_metadata_batch(user, access_token, body)


@router.post(
    "/predictions/{prediction_id}/image",
    response_model=PredictionImageUploadOut,
    responses=_PREDICT_RESPONSES,
    tags=["offline-sync"],
    summary="Sync offline — paso 2: imagen",
    description="Multipart ``image``; ``image_sha256`` opcional (409 si no coincide).",
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
