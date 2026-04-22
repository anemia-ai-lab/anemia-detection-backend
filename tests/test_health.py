from fastapi.testclient import TestClient

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
