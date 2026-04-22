"""Códigos de error HTTP por defecto cuando el dominio no aporta ``code``."""

from __future__ import annotations


def default_error_code(status_code: int) -> str:
    mapping: dict[int, str] = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
        502: "bad_gateway",
        503: "service_unavailable",
    }
    return mapping.get(status_code, "error")
