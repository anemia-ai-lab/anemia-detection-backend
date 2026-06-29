"""Tests de verificación JWT local opcional."""

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from backend.core.jwt_verifier import (
    JwtVerificationError,
    verify_access_token_local,
)


def test_verify_access_token_local_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "test-jwt-secret-for-unit-tests-32bytes!"
    monkeypatch.setattr("backend.core.jwt_verifier.settings.supabase_jwt_secret", secret)
    exp = datetime.now(UTC) + timedelta(hours=1)
    token = jwt.encode(
        {"sub": "user-123", "email": "u@example.com", "exp": exp},
        secret,
        algorithm="HS256",
    )
    user = verify_access_token_local(token)
    assert user.id == "user-123"
    assert user.email == "u@example.com"


def test_verify_access_token_local_invalid_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("backend.core.jwt_verifier.settings.supabase_jwt_secret", "secret-a-for-jwt-unit-tests-32b!")
    exp = datetime.now(UTC) + timedelta(hours=1)
    token = jwt.encode(
        {"sub": "user-123", "exp": exp},
        "secret-b-for-jwt-unit-tests-32b!",
        algorithm="HS256",
    )
    with pytest.raises(JwtVerificationError) as exc:
        verify_access_token_local(token)
    assert exc.value.code == "invalid_token"
