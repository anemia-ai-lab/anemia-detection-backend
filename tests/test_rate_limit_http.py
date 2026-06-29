"""Rate limit HTTP 429 en rutas protegidas."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from backend.api import deps as api_deps
from backend.core import config as config_module
from backend.main import app
from backend.schemas.auth import UserOut
from backend.services.prediction_service import PredictionService

client = TestClient(app)


def _minimal_png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(200, 160, 140)).save(buf, format="PNG")
    return buf.getvalue()


def test_predict_returns_429_when_rate_limit_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    reset = getattr(app.state, "rate_limit_reset_buckets", None)
    if callable(reset):
        reset()
    monkeypatch.setattr(config_module.settings, "rate_limit_enabled", True)
    monkeypatch.setattr(config_module.settings, "rate_limit_predict_requests", 2)
    monkeypatch.setattr(config_module.settings, "rate_limit_window_seconds", 60)

    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="u@example.com",
        created_at=None,
    )

    def fake_ctx() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    class _StubSvc(PredictionService):
        async def run_predict_from_upload(self, *args, **kwargs):  # noqa: ANN002, ANN003
            from backend.schemas.prediction import PredictionResponse

            return PredictionResponse.model_validate(
                {
                    "id": "33333333-3333-3333-3333-333333333333",
                    "risk": "low",
                    "score": 0.1,
                    "raw_probability": 0.1,
                    "calibrated_probability": 0.1,
                    "threshold_used": 0.168,
                    "prediction": 0,
                    "risk_label": "Low anemia risk prediction",
                    "model_version": "v2.0",
                    "created_at": "2026-04-01T12:00:00Z",
                    "inference_mode": "backend",
                },
            )

    app.dependency_overrides[api_deps.get_predict_context] = fake_ctx
    app.dependency_overrides[api_deps.get_prediction_service] = lambda: _StubSvc()
    try:
        headers = {"Authorization": "Bearer aaa.bbb.ccc"}
        files = {"image": ("m.png", _minimal_png(), "image/png")}
        for _ in range(2):
            r = client.post("/predict", headers=headers, files=files)
            assert r.status_code == 200
        r3 = client.post("/predict", headers=headers, files=files)
        assert r3.status_code == 429
        assert r3.json()["code"] == "rate_limit_exceeded"
    finally:
        app.dependency_overrides.clear()


def test_sync_metadata_returns_429_when_rate_limit_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset = getattr(app.state, "rate_limit_reset_buckets", None)
    if callable(reset):
        reset()
    monkeypatch.setattr(config_module.settings, "rate_limit_enabled", True)
    monkeypatch.setattr(config_module.settings, "rate_limit_sync_metadata_requests", 2)
    monkeypatch.setattr(config_module.settings, "rate_limit_window_seconds", 60)

    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="u@example.com",
        created_at=None,
    )

    def fake_ctx() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    class _StubSvc(PredictionService):
        def sync_metadata_batch(self, *args, **kwargs):  # noqa: ANN002, ANN003
            from backend.schemas.prediction import PredictionSyncMetadataResponse

            return PredictionSyncMetadataResponse(results=[])

    app.dependency_overrides[api_deps.get_predict_context] = fake_ctx
    app.dependency_overrides[api_deps.get_prediction_service] = lambda: _StubSvc()
    payload = {"items": []}
    try:
        headers = {"Authorization": "Bearer aaa.bbb.ccc"}
        for _ in range(2):
            r = client.post("/predictions/sync/metadata", headers=headers, json=payload)
            assert r.status_code == 422  # empty items validation
        r3 = client.post("/predictions/sync/metadata", headers=headers, json=payload)
        assert r3.status_code == 429
    finally:
        app.dependency_overrides.clear()
