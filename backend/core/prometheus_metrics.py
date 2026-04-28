"""Métricas Prometheus (agregados con etiquetas de baja cardinalidad)."""

from __future__ import annotations

import re
import time

from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.registry import REGISTRY
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from backend.inference.runtime import inference_service_status

# Formato texto Prometheus clásico (evitar OpenMetrics 1.0 por defecto en algunas versiones).
PROMETHEUS_TEXT_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests procesadas.",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Duración HTTP hasta respuesta (segundos).",
    ["method", "path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, float("inf")),
)

PREDICTIONS_COMPLETED_TOTAL = Counter(
    "predictions_completed_total",
    "Predicciones completadas con éxito (POST /predict, HTTP < 400).",
)

PREDICTION_ERRORS_TOTAL = Counter(
    "prediction_errors_total",
    "Fallos en POST /predict (HTTP >= 400 o excepción no tratada).",
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "1 si el predictor Keras embarcado está cargado; 0 si no.",
)


# Rutas estáticas conocidas (coincidencia exacta tras normalización ligera).
_STATIC_ROUTE_TEMPLATES: frozenset[str] = frozenset(
    {
        "/",
        "/health",
        "/metrics",
        "/predict",
        "/predictions",
        "/auth/register",
        "/auth/login",
        "/auth/me",
        "/auth/me/profile",
        "/model/evaluation",
        "/openapi.json",
        "/docs",
        "/redoc",
    },
)


_DYNAMIC_ROUTE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"^/predictions/[^/]+/image-signed-url$"),
        "/predictions/{id}/image-signed-url",
    ),
)


def _strip_trailing_slash_except_root(path: str) -> str:
    if path == "/" or len(path) <= 1:
        return path or "/"
    return path.rstrip("/") or "/"


def route_template_for_path(path: str) -> str:
    """
    Convierte la ruta HTTP a una **plantilla fija** (baja cardinalidad).

    Ignora query y fragmento si vinieran en el string. No incorpora firmas ni nombres de fichero;
    los segmentos variables usan plantillas explícitas.
    """
    p0 = path.split("?", 1)[0].split("#", 1)[0]
    p = _strip_trailing_slash_except_root(p0)
    if p in _STATIC_ROUTE_TEMPLATES:
        return p
    for pattern, template in _DYNAMIC_ROUTE_RULES:
        if pattern.match(p):
            return template
    return "/other"


def _path_label_for_request(request: Request) -> str:
    """Plantilla de ruta sólo desde ``path`` normalizado (sin query ni fragmentos)."""
    return route_template_for_path(request.url.path)


def build_metrics_response() -> tuple[bytes, str]:
    """
    Cuerpo Prometheus (texto 0.0.4) y ``Content-Type`` explícito.

    ``model_loaded`` se recalcula en cada *scrape* (estado real vía ``inference_service_status``).
    """
    _svc, loaded = inference_service_status()
    MODEL_LOADED.set(1.0 if loaded else 0.0)

    from prometheus_client import generate_latest  # noqa: PLC0415

    return generate_latest(REGISTRY), PROMETHEUS_TEXT_CONTENT_TYPE


def register_prometheus_middleware(app: FastAPI) -> None:
    """Instrumentación HTTP ligera; registrar como último middleware."""

    class _PrometheusHTTPMiddleware(BaseHTTPMiddleware):
        async def dispatch(
            self,
            request: Request,
            call_next: RequestResponseEndpoint,
        ) -> Response:
            start = time.perf_counter()
            method = request.method
            raw_path = request.url.path
            resp: Response | None = None
            status_code = 500
            norm_predict = _strip_trailing_slash_except_root(raw_path)
            predict_post = method == "POST" and norm_predict == "/predict"
            try:
                resp = await call_next(request)
                status_code = int(resp.status_code)
                return resp
            except Exception:
                status_code = 500
                raise
            finally:
                elapsed = time.perf_counter() - start
                path_label = _path_label_for_request(request)
                sc = str(status_code)
                HTTP_REQUESTS_TOTAL.labels(
                    method=method,
                    path=path_label,
                    status=sc,
                ).inc()
                HTTP_REQUEST_DURATION_SECONDS.labels(
                    method=method,
                    path=path_label,
                ).observe(elapsed)
                if predict_post:
                    if status_code < 400:
                        PREDICTIONS_COMPLETED_TOTAL.inc()
                    else:
                        PREDICTION_ERRORS_TOTAL.inc()

    app.add_middleware(_PrometheusHTTPMiddleware)
