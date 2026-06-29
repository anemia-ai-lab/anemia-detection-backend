"""Shared test helpers (no pytest plugin hooks required)."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from supabase_auth.types import AuthResponse, Session, User, UserResponse

from backend.core import config as config_module

# La suite del backend no debe inicializar TensorFlow/Keras (velocidad, determinismo, sin exit 134).
# Opcional: ALLOW_BACKEND_TF=1 para cargar el .keras real durante tests (no recomendado en CI).
if os.environ.get("ALLOW_BACKEND_TF", "").strip().lower() not in ("1", "true", "yes", "on"):
    os.environ["DISABLE_TF"] = "1"
    os.environ["INFERENCE_MODEL_PATH"] = ""
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("PREDICT_MULTINAIL_ENABLED", "false")


@pytest.fixture(autouse=True)
def _disable_multinail_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """La suite API usa modo legacy salvo tests que habiliten multinail explícitamente."""
    monkeypatch.setattr(config_module.settings, "predict_multinail_enabled", False)


@pytest.fixture(autouse=True)
def _disable_rate_limit_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evita contaminar el bucket en memoria entre tests de /predict."""
    monkeypatch.setattr(config_module.settings, "rate_limit_enabled", False)


def dt() -> datetime:
    return datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)


def sample_user(*, email: str = "user@example.com") -> User:
    t = dt()
    return User(
        id="11111111-1111-1111-1111-111111111111",
        app_metadata={},
        user_metadata={"role_hint": "patient"},
        aud="authenticated",
        email=email,
        created_at=t,
        updated_at=t,
    )


def sample_session(user: User) -> Session:
    return Session(
        access_token="aaa.bbb.ccc",
        refresh_token="refresh-token-value",
        expires_in=3600,
        token_type="bearer",
        user=user,
    )


def sample_auth_response(
    *,
    with_session: bool = True,
    email: str = "user@example.com",
) -> AuthResponse:
    user = sample_user(email=email)
    session = sample_session(user) if with_session else None
    return AuthResponse(user=user, session=session)


def sample_user_response(*, email: str = "user@example.com") -> UserResponse:
    return UserResponse(user=sample_user(email=email))
