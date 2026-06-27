"""Validación de Settings en producción."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.core.config import Settings


def test_production_requires_supabase_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_KEY", "")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "")
    monkeypatch.setenv("METRICS_BEARER_TOKEN", "")
    with pytest.raises(ValidationError) as exc:
        Settings()
    msg = str(exc.value)
    assert "SUPABASE_URL" in msg
    assert "METRICS_BEARER_TOKEN" in msg


def test_production_rejects_placeholder_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("SUPABASE_URL", "REPLACE_ME")
    monkeypatch.setenv("SUPABASE_KEY", "REPLACE_ME")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "REPLACE_ME")
    monkeypatch.setenv("METRICS_BEARER_TOKEN", "REPLACE_ME")
    with pytest.raises(ValidationError) as exc:
        Settings()
    msg = str(exc.value)
    assert "REPLACE_ME" in msg
    assert "SUPABASE_URL" in msg


def test_production_rejects_non_default_storage_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "anon-key")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("METRICS_BEARER_TOKEN", "metrics-secret")
    monkeypatch.setenv("PREDICTIONS_STORAGE_BUCKET", "other-bucket")
    with pytest.raises(ValidationError) as exc:
        Settings()
    assert "PREDICTIONS_STORAGE_BUCKET" in str(exc.value)
