from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from backend.core import config as config_module
from backend.main import app

client = TestClient(app)


def test_model_evaluation_returns_config_metrics() -> None:
    response = client.get("/model/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "v1.0"
    assert data["sensitivity"] == 0.82
    assert data["specificity"] == 0.79
    assert data["f1_score"] == 0.8
    assert data["auc_roc"] == 0.88
    assert data["inference_time_ms"] == 45.0
    assert data["model_size_mb"] == 12.5
    assert data["grad_cam_available"] is True
    assert data["dataset_version"] == "internal-v1"
    assert data["evaluated_at"] in (
        "2026-01-15T12:00:00Z",
        "2026-01-15T12:00:00+00:00",
    )


def test_model_evaluation_reflects_settings_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.schemas.model_evaluation import ModelEvalMetrics

    custom = ModelEvalMetrics(
        sensitivity=0.5,
        specificity=0.5,
        f1_score=0.5,
        auc_roc=0.5,
        inference_time_ms=1.0,
        model_size_mb=2.0,
        grad_cam_available=False,
        evaluated_at=datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC),
        dataset_version="unit-test-ds",
    )
    monkeypatch.setattr(config_module.settings, "model_eval", custom)
    monkeypatch.setattr(config_module.settings, "model_version", "test-9.9.9")
    response = client.get("/model/evaluation")
    assert response.status_code == 200
    assert response.json()["model_version"] == "test-9.9.9"
    assert response.json()["grad_cam_available"] is False
    assert response.json()["dataset_version"] == "unit-test-ds"
