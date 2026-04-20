"""Bearer + JWT shape checks only.

Authoritative verification of the access token happens in Supabase (``get_user``).
"""

from __future__ import annotations

from typing import Optional, Protocol


class _BearerCredentials(Protocol):
    scheme: str
    credentials: str


class TokenValidationError(Exception):
    """Invalid or missing bearer token before calling Supabase."""

    def __init__(self, message: str, *, code: str) -> None:
        self.message = message
        self.code = code
        super().__init__(message)


def parse_bearer_token(credentials: Optional[_BearerCredentials]) -> str:
    if credentials is None:
        raise TokenValidationError("Not authenticated", code="missing_token")
    if credentials.scheme.lower() != "bearer":
        raise TokenValidationError(
            "Invalid authorization scheme",
            code="invalid_scheme",
        )
    token = credentials.credentials.strip()
    if not token:
        raise TokenValidationError("Not authenticated", code="empty_token")
    return token


def require_well_formed_jwt(access_token: str) -> None:
    """Three non-empty segments (Supabase access tokens are JWTs)."""
    parts = access_token.split(".")
    if len(parts) != 3 or not all(p.strip() for p in parts):
        raise TokenValidationError("Invalid access token", code="malformed_token")
