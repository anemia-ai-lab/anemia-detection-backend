"""Re-exportación de excepciones HTTP compartidas (:mod:`backend.core.exceptions`)."""

from backend.core.exceptions import AuthServiceError, ClientHttpError, PredictionServiceError

__all__ = ["AuthServiceError", "ClientHttpError", "PredictionServiceError"]
