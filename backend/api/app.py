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
from backend.core.config import settings
from backend.core.http_error_codes import default_error_code
from backend.core.logging_config import configure_logging
from backend.core.prometheus_metrics import build_metrics_response, register_prometheus_middleware
from backend.core.rate_limit import register_rate_limit_middleware
from backend.inference.runtime import (
    get_builtin_image_predictor,
    inference_service_status,
    init_inference_model,
    shutdown_inference_model,
)
from backend.schemas.health import HealthOut
from backend.services.exceptions import ClientHttpError

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
    logger.info("inference_model_ready=%s", get_builtin_image_predictor() is not None)
    yield
    shutdown_inference_model()
    logger.info("application_shutdown")


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "health",
            "description": (
                "Disponibilidad del proceso y del modelo de inferencia (sin secretos). "
                "Útil para balanceadores, despliegue y demos; no implica validación clínica."
            ),
        },
        {
            "name": "auth",
            "description": (
                "Supabase Auth: registro, login, /me, perfil (GET/PATCH). "
                "El email del perfil sale de auth, no de la tabla."
            ),
        },
        {
            "name": "predictions",
            "description": (
                "Predicción de riesgo con imagen (Keras si el modelo está cargado), historial y URLs firmadas. "
                "Salidas cuantitativas y binarias asistidas por modelo; no diagnóstico."
            ),
        },
        {
            "name": "model",
            "description": (
                "Métricas de evaluación offline fijadas en configuración (trazabilidad tesis/paper). "
                "Sin inferencia en vivo; no sustituye informe clínico."
            ),
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
        content={"detail": exc.message, "code": code},
    )


@app.get(
    "/health",
    tags=["health"],
    summary="Estado del API y de la inferencia",
    description=(
        "Comprueba que el proceso responde y expone si el predictor Keras está cargado, la versión de modelo "
        "declarada, si la calibración por temperatura es no trivial (T≠1) y la ruta configurada del artefacto "
        "(sin rutas absolutas arbitrarias ni credenciales). No evalúa calidad clínica del despliegue."
    ),
    response_model=HealthOut,
    response_model_exclude_none=True,
)
def health() -> HealthOut:
    svc_status, model_loaded = inference_service_status()
    calibration_enabled = abs(float(settings.inference_calibration_temperature) - 1.0) > 1e-12
    raw_path = settings.inference_model_path.strip()
    return HealthOut(
        status=svc_status,
        model_loaded=model_loaded,
        model_version=settings.model_version,
        calibration_enabled=calibration_enabled,
        inference_model_path=(raw_path or None) if _local_environment() else None,
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


app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(model_router, prefix="/model")
app.include_router(predict_router)

register_prometheus_middleware(app)
register_rate_limit_middleware(app)
