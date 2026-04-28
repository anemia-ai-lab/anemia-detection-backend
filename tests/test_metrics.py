"""Smoke tests para ``/metrics`` (Prometheus)."""

from fastapi.testclient import TestClient

from backend.core.prometheus_metrics import route_template_for_path
from backend.main import app

client = TestClient(app)


def test_metrics_endpoint_returns_prometheus_text() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "http_requests_total" in body
    assert "http_request_duration_seconds" in body
    assert "predictions_completed_total" in body
    assert "prediction_errors_total" in body
    assert "model_loaded" in body
    ctype = response.headers.get("content-type", "")
    assert ctype.startswith("text/plain")
    assert "version=0.0.4" in ctype


def test_route_templates_low_cardinality() -> None:
    assert route_template_for_path("/health") == "/health"
    pid = "550e8400-e29b-41d4-a716-446655440000"
    assert route_template_for_path(f"/predictions/{pid}/image-signed-url") == (
        "/predictions/{id}/image-signed-url"
    )
    assert route_template_for_path("/predict?q=1") == "/predict"
    weird = "/predictions/evil/../../../etc/passwd/image-signed-url"
    assert route_template_for_path(weird) == "/other"


def test_health_unchanged_fields() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) >= {"status", "model_loaded", "model_version"}
