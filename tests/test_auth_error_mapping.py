"""Mapeo de errores Supabase Auth a AuthServiceError."""

from __future__ import annotations

from supabase_auth.errors import AuthApiError

from backend.services.auth_error_mapping import map_supabase_auth_error


def test_maps_429_to_rate_limit_message() -> None:
    exc = AuthApiError("too many", 429, "over_request_rate_limit")
    err = map_supabase_auth_error(exc)
    assert err.status_code == 429
    assert "rate limit" in err.detail.lower()


def test_maps_credentials_to_safe_message_with_prefer_unauthorized() -> None:
    exc = AuthApiError("Invalid login credentials", 400, "invalid_credentials")
    err = map_supabase_auth_error(exc, prefer_unauthorized=True)
    assert err.status_code == 401
    assert "password" in err.detail.lower() or "email" in err.detail.lower()
