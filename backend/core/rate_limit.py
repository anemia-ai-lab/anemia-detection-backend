"""Rate limiting ligero en memoria para rutas sensibles."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from backend.core.config import settings

_Bucket = Deque[float]


def rate_limit_client_key(request: Request) -> str:
    """Identificador estable por petición para ventanas de rate limit (preferencia: socket)."""
    if settings.trust_proxy_headers:
        forwarded_for = request.headers.get("x-forwarded-for", "")
        if forwarded_for:
            return forwarded_for.split(",", 1)[0].strip()
    if request.client is not None:
        return request.client.host
    return "unknown"


def _route_limit(path: str, method: str) -> int | None:
    if method != "POST":
        return None
    normalized = path.rstrip("/") or "/"
    if normalized in {"/auth/login", "/auth/register"}:
        return int(settings.rate_limit_auth_requests)
    if normalized == "/predict":
        return int(settings.rate_limit_predict_requests)
    return None


def register_rate_limit_middleware(app: FastAPI) -> None:
    """Registra límites por cliente y ruta en una ventana móvil simple."""

    buckets: defaultdict[tuple[str, str], _Bucket] = defaultdict(deque)

    class _RateLimitMiddleware(BaseHTTPMiddleware):
        async def dispatch(
            self,
            request: Request,
            call_next: RequestResponseEndpoint,
        ) -> Response:
            if not settings.rate_limit_enabled:
                return await call_next(request)
            limit = _route_limit(request.url.path, request.method)
            if limit is None:
                return await call_next(request)

            now = time.monotonic()
            window = float(settings.rate_limit_window_seconds)
            key = (rate_limit_client_key(request), request.url.path.rstrip("/") or "/")
            bucket = buckets[key]
            while bucket and now - bucket[0] >= window:
                bucket.popleft()
            if len(bucket) >= limit:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Too many requests. Please retry later.",
                        "code": "rate_limit_exceeded",
                    },
                    headers={"Retry-After": str(int(window))},
                )
            bucket.append(now)
            return await call_next(request)

    app.add_middleware(_RateLimitMiddleware)
