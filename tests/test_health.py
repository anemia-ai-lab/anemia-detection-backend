from fastapi.testclient import TestClient

from backend.core import config as config_module
from backend.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "model_loaded" in data
    assert data["model_version"] == "v1.0"
    assert "calibration_enabled" in data
    assert isinstance(data["calibration_enabled"], bool)
    if data.get("inference_model_path"):
        assert isinstance(data["inference_model_path"], str)


def test_health_omits_model_path_outside_local(monkeypatch) -> None:
    monkeypatch.setattr(config_module.settings, "environment", "production")
    monkeypatch.setattr(
        config_module.settings,
        "inference_model_path",
        "ml/artifacts/models/baseline_mobilenetv2.keras",
    )
    response = client.get("/health")
    assert response.status_code == 200
    assert "inference_model_path" not in response.json()
