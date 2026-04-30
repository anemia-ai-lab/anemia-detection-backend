"""Map Supabase GoTrue errors to :class:`AuthServiceError` (no HTTP types here)."""

from supabase_auth.errors import AuthApiError, AuthError, AuthUnknownError, CustomAuthError

from backend.services.exceptions import AuthServiceError


def _go_true_code(exc: AuthError) -> str | None:
    if exc.code is None:
        return None
    return str(exc.code)


def _safe_auth_message(
    *,
    code: str | None,
    status: int,
    source_message: str,
) -> str:
    lower = f"{code or ''} {source_message}".lower()
    if status == 429:
        return "Authentication rate limit exceeded"
    if "credential" in lower or "password" in lower or "login" in lower:
        return "Invalid email or password"
    if "jwt" in lower or "token" in lower:
        return "Invalid authentication token"
    if "already" in lower or "exists" in lower or status == 409:
        return "Registration conflict"
    if status >= 500:
        return "Authentication service error"
    return "Authentication request failed"


def map_supabase_auth_error(
    exc: AuthError,
    *,
    prefer_unauthorized: bool = False,
) -> AuthServiceError:
    code = _go_true_code(exc)

    if isinstance(exc, AuthApiError):
        status = exc.status if exc.status >= 400 else 400
        source_message = exc.message or ""
        if prefer_unauthorized and status in (400, 401, 403, 422):
            lower = f"{code or ''} {source_message}".lower()
            if any(
                part in lower
                for part in ("invalid", "credential", "wrong", "password", "jwt")
            ):
                status = 401
        message = _safe_auth_message(
            code=code,
            status=status,
            source_message=source_message,
        )
        return AuthServiceError(message, status, code=code)
    if isinstance(exc, CustomAuthError):
        status = exc.status if exc.status >= 400 else 400
        return AuthServiceError(
            _safe_auth_message(code=code, status=status, source_message=exc.message or ""),
            status,
            code=code,
        )
    if isinstance(exc, AuthUnknownError):
        return AuthServiceError(
            "Authentication service error",
            502,
            code="auth_upstream",
        )
    if isinstance(exc, AuthError):
        return AuthServiceError(
            _safe_auth_message(code=code, status=400, source_message=exc.message or ""),
            400,
            code=code,
        )
    return AuthServiceError(
        "Authentication service error",
        500,
        code="auth_internal",
    )
