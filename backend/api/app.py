import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse, Response

from backend.api.routes.auth import router as auth_router
from backend.api.routes.model import router as model_router
from backend.api.routes.predict import router as predict_router
from backend.core.config import settings, supabase_config_ready
from backend.core.exceptions import ClientHttpError
from backend.core.http_error_codes import default_error_code
from backend.core.logging_config import configure_logging
from backend.core.prometheus_metrics import build_metrics_response, register_prometheus_middleware
from backend.core.rate_limit import register_rate_limit_middleware
from backend.inference.nail_detection import (
    get_cached_hand_landmarker_status,
    shutdown_hand_landmarker,
)
from backend.inference.runtime import (
    get_builtin_image_predictor,
    inference_service_status,
    init_inference_model,
    shutdown_inference_model,
    warmup_inference_model,
)
from backend.schemas.health import HealthOut

configure_logging()
logger = logging.getLogger(__name__)

_LOCAL_ENVIRONMENTS = {"development", "dev", "local", "test", "testing"}


def _prediction_api_path(path: str) -> bool:
    return path == "/predict" or path.startswith("/predictions")


def _local_environment() -> bool:
    return settings.environment.strip().lower() in _LOCAL_ENVIRONMENTS


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(
        "application_startup environment=%s debug=%s",
        settings.environment,
        settings.debug,
    )
    init_inference_model()
    warmup_inference_model()
    logger.info("inference_model_ready=%s", get_builtin_image_predictor() is not None)
    if settings.predict_multinail_enabled:
        lm_status = get_cached_hand_landmarker_status(refresh=True)
        logger.info(
            "hand_landmarker_ready=%s model_exists=%s error=%s",
            lm_status.get("ready"),
            lm_status.get("model_exists"),
            lm_status.get("error"),
        )
    yield
    shutdown_inference_model()
    shutdown_hand_landmarker()
    logger.info("application_shutdown")


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "health",
            "description": "Disponibilidad del proceso y del modelo Keras cargado.",
        },
        {
            "name": "auth",
            "description": (
                "Supabase Auth: registro, login, refresh (opcional; supabase-js puede refrescar directo), /me."
            ),
        },
        {
            "name": "profile",
            "description": "Tabla profiles: GET/PATCH /auth/me/profile. Email desde auth.",
        },
        {
            "name": "predictions",
            "description": "Inferencia online e historial CRUD del usuario autenticado.",
        },
        {
            "name": "offline-sync",
            "description": "Sync TFLite móvil: metadatos en batch + imagen. Idempotente por client_id.",
        },
        {
            "name": "model",
            "description": "Métricas de evaluación offline en configuración (model_version).",
        },
    ],
)

_cors_origins = settings.effective_cors_origins()
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    errors = exc.errors()
    if errors:
        err0 = errors[0]
        loc_parts = [str(x) for x in err0.get("loc", ()) if x not in ("body", "query", "path")]
        loc = ".".join(loc_parts) if loc_parts else ""
        msg = str(err0.get("msg", "Validation error"))
        detail = f"{msg}" + (f" ({loc})" if loc else "")
    else:
        detail = "Validation error"
    return JSONResponse(
        status_code=422,
        content={"detail": detail, "code": "validation_error"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    detail_str = detail if isinstance(detail, str) else str(detail)
    code = default_error_code(exc.status_code)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": detail_str, "code": code},
    )


@app.exception_handler(ClientHttpError)
async def client_http_error_handler(
    request: Request,
    exc: ClientHttpError,
) -> JSONResponse:
    code = exc.code or default_error_code(exc.status_code)
    if _prediction_api_path(request.url.path):
        log_msg = "predict_request_failed status=%s code=%s"
        args = (exc.status_code, code)
        if exc.status_code >= 500:
            logger.error(log_msg, *args)
        else:
            logger.warning(log_msg, *args)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": code},
    )


@app.get(
    "/health",
    tags=["health"],
    summary="Estado del API y de la inferencia",
    description="model_loaded, model_version, calibration_enabled, supabase_ready.",
    response_model=HealthOut,
    response_model_exclude_none=True,
)
def health() -> HealthOut:
    svc_status, model_loaded = inference_service_status()
    calibration_enabled = abs(float(settings.inference_calibration_temperature) - 1.0) > 1e-12
    raw_path = settings.inference_model_path.strip()
    lm_ready: bool | None = None
    lm_error: str | None = None
    if settings.predict_multinail_enabled:
        lm_status = get_cached_hand_landmarker_status()
        lm_ready = bool(lm_status.get("ready"))
        err = lm_status.get("error")
        lm_error = str(err)[:200] if err else None
    return HealthOut(
        status=svc_status,
        model_loaded=model_loaded,
        model_version=settings.model_version,
        calibration_enabled=calibration_enabled,
        supabase_ready=supabase_config_ready(),
        inference_model_path=(raw_path or None) if _local_environment() else None,
        hand_landmarker_ready=lm_ready,
        hand_landmarker_error=lm_error,
    )


@app.get(
    "/metrics",
    summary="Prometheus (métricas internas)",
    description=(
        "Formato Prometheus/OpenMetrics: contadores HTTP, latencias, métricas de ``POST /predict`` "
        "y estado ``model_loaded``. Sin datos personales ni tokens."
    ),
    include_in_schema=False,
)
def prometheus_metrics_endpoint(request: Request) -> Response:
    token = settings.metrics_bearer_token.strip()
    if not _local_environment():
        if not token:
            raise HTTPException(status_code=404, detail="Not found")
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {token}":
            raise HTTPException(status_code=403, detail="Forbidden")
    payload, ctype = build_metrics_response()
    return Response(content=payload, media_type=ctype)


app.include_router(auth_router, prefix="/auth")
app.include_router(model_router, prefix="/model")
app.include_router(predict_router)

register_prometheus_middleware(app)
register_rate_limit_middleware(app)
