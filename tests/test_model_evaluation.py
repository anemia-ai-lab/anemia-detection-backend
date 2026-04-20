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
    assert data["auc"] == pytest.approx(0.795092)
    assert data["precision_operational"] == pytest.approx(0.454545)
    assert data["recall_operational"] == pytest.approx(0.740741)
    assert data["accuracy_operational"] == pytest.approx(0.793333)
    assert data["operational_threshold"] == pytest.approx(0.1680544387290045)
    assert data["temperature"] == pytest.approx(0.7510018331928743)
    assert data["brier_score"] == pytest.approx(0.11766287029947034)
    assert data["expected_calibration_error"] == pytest.approx(0.060344067420365466)
    assert data["oversampling_used"] is True
    assert data["class_weight_used"] is False
    assert data["fine_tuning_used"] is True
    assert data["dataset_version"] == "experiment_20260420T043804Z; calibration_20260420T045056Z"
    assert data["evaluated_at"].startswith("2026-04-20T04:50:56")


def test_model_evaluation_reflects_settings_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.schemas.model_evaluation import ModelEvalMetrics

    custom = ModelEvalMetrics(
        auc=0.5,
        precision_operational=0.4,
        recall_operational=0.6,
        accuracy_operational=0.7,
        operational_threshold=0.2,
        temperature=1.0,
        brier_score=0.2,
        expected_calibration_error=0.1,
        oversampling_used=False,
        class_weight_used=True,
        fine_tuning_used=False,
        evaluated_at=datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC),
        dataset_version="unit-test-ds",
    )
    monkeypatch.setattr(config_module.settings, "model_eval", custom)
    monkeypatch.setattr(config_module.settings, "model_version", "test-9.9.9")
    response = client.get("/model/evaluation")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test-9.9.9"
    assert body["auc"] == 0.5
    assert body["oversampling_used"] is False
    assert body["class_weight_used"] is True
    assert body["fine_tuning_used"] is False
    assert body["dataset_version"] == "unit-test-ds"
