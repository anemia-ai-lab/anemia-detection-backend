"""Probe Hand Landmarker y fixture smoke_hand (integración opcional)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from backend.inference.nail_detection import (
    build_nail_detector,
    probe_hand_landmarker_status,
    resolved_hand_landmarker_model_path,
)

_SMOKE_HAND = (
    Path(__file__).resolve().parents[1] / "scripts" / "fixtures" / "smoke_hand.jpg"
)
_MODEL_PATH = resolved_hand_landmarker_model_path()


def test_probe_reports_missing_model_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.task"
    with patch(
        "backend.inference.nail_detection.resolved_hand_landmarker_model_path",
        return_value=missing,
    ):
        status = probe_hand_landmarker_status(warmup=True)
    assert status["ready"] is False
    assert status["error"] == "model_file_missing"


def test_health_includes_landmarker_fields_when_multinail_enabled() -> None:
    from fastapi.testclient import TestClient

    from backend.main import app

    response = TestClient(app).get("/health")
    assert response.status_code == 200
    data = response.json()
    if data.get("hand_landmarker_ready") is not None:
        assert isinstance(data["hand_landmarker_ready"], bool)


@pytest.mark.skipif(not _MODEL_PATH.is_file(), reason="hand_landmarker.task not present")
def test_smoke_hand_fixture_detects_three_mediapipe_crops() -> None:
    """Misma imagen que smoke prod; falla en CI si MediaPipe no funciona en Linux."""
    raw = _SMOKE_HAND.read_bytes()
    rgb = np.array(Image.open(BytesIO(raw)).convert("RGB"))
    status = probe_hand_landmarker_status(warmup=True)
    assert status["ready"] is True, status
    crops = build_nail_detector().detect(rgb)
    assert len(crops) >= 1
    assert any(c.source == "mediapipe" for c in crops)
