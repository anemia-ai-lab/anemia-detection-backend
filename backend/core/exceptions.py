"""Errores orientados a cliente HTTP (detalle + código opcional); capas inferiores pueden usar estos tipos sin depender de ``backend.services``."""


class ClientHttpError(Exception):
    """Error de dominio mapeado a respuestas JSON ``{"detail", "code"}``."""

    def __init__(
        self,
        detail: str,
        status_code: int = 400,
        *,
        code: str | None = None,
    ) -> None:
        self.detail = detail
        self.status_code = status_code
        self.code = code
        super().__init__(detail)


class AuthServiceError(ClientHttpError):
    """Flujos de autenticación y token."""


class PredictionServiceError(ClientHttpError):
    """Validación de predicción, persistencia o almacenamiento expuesto al cliente."""
