"""Map Supabase GoTrue errors to :class:`AuthServiceError` (no HTTP types here)."""

from supabase_auth.errors import AuthApiError, AuthError, AuthUnknownError, CustomAuthError

from backend.services.exceptions import AuthServiceError


def _go_true_code(exc: AuthError) -> str | None:
    if exc.code is None:
        return None
    return str(exc.code)


def map_supabase_auth_error(
    exc: AuthError,
    *,
    prefer_unauthorized: bool = False,
) -> AuthServiceError:
    code = _go_true_code(exc)

    if isinstance(exc, AuthApiError):
        status = exc.status if exc.status >= 400 else 400
        message = exc.message
        if prefer_unauthorized and status in (400, 401, 403, 422):
            lower = message.lower()
            if any(
                part in lower
                for part in ("invalid", "credential", "wrong", "password", "jwt")
            ):
                status = 401
        return AuthServiceError(message, status, code=code)
    if isinstance(exc, CustomAuthError):
        status = exc.status if exc.status >= 400 else 400
        return AuthServiceError(exc.message, status, code=code)
    if isinstance(exc, AuthUnknownError):
        return AuthServiceError(
            "Authentication service error",
            502,
            code="auth_upstream",
        )
    if isinstance(exc, AuthError):
        return AuthServiceError(exc.message, 400, code=code)
    return AuthServiceError(
        "Authentication service error",
        500,
        code="auth_internal",
    )
