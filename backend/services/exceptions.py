class ClientHttpError(Exception):
    """Domain error mapped to JSON HTTP responses (detail + optional code)."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        *,
        code: str | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.code = code
        super().__init__(message)


class AuthServiceError(ClientHttpError):
    """Auth / token flows."""


class PredictionServiceError(ClientHttpError):
    """Prediction persistence or validation."""
