"""Verificación local opcional de JWT Supabase (HS256)."""

from __future__ import annotations

from dataclasses import dataclass

import jwt

from backend.core.config import settings


class JwtVerificationError(Exception):
    """JWT inválido o no verificable localmente."""

    def __init__(self, message: str, *, code: str) -> None:
        self.message = message
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class VerifiedJwtUser:
    id: str
    email: str | None


def jwt_local_verification_enabled() -> bool:
    return bool(settings.supabase_jwt_secret.strip())


def verify_access_token_local(access_token: str) -> VerifiedJwtUser:
    """Verifica firma y expiración; extrae ``sub`` y ``email`` del payload."""
    secret = settings.supabase_jwt_secret.strip()
    if not secret:
        msg = "SUPABASE_JWT_SECRET not configured"
        raise JwtVerificationError(msg, code="jwt_secret_missing")
    try:
        payload = jwt.decode(
            access_token,
            secret,
            algorithms=["HS256"],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise JwtVerificationError("Token expired", code="token_expired") from exc
    except jwt.InvalidTokenError as exc:
        raise JwtVerificationError("Invalid access token", code="invalid_token") from exc

    sub = payload.get("sub")
    if not sub or not str(sub).strip():
        raise JwtVerificationError("Invalid access token", code="invalid_token")
    email_raw = payload.get("email")
    email = str(email_raw).strip() if email_raw else None
    return VerifiedJwtUser(id=str(sub), email=email)
