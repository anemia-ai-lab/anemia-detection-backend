"""Tests de validación de metadatos sync offline."""

from datetime import UTC, datetime

import pytest

from backend.core.config import settings
from backend.core.exceptions import ClientHttpError
from backend.schemas.prediction import PredictionSyncMetadataItem
from backend.services.sync_metadata_validation import validate_sync_metadata_item


def _valid_item(**overrides: object) -> PredictionSyncMetadataItem:
    payload = {
        "client_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "risk": "low",
        "score": 0.12,
        "raw_probability": 0.15,
        "calibrated_probability": 0.12,
        "threshold_used": float(settings.inference_risk_tier_high_lower),
        "prediction": 0,
        "model_version": settings.model_version,
        "inference_mode": "tflite_offline",
        "client_created_at": datetime(2026, 4, 30, 8, 0, 0, tzinfo=UTC),
    }
    payload.update(overrides)
    return PredictionSyncMetadataItem.model_validate(payload)


def test_validate_sync_metadata_accepts_coherent_payload() -> None:
    item = _valid_item()
    validate_sync_metadata_item(
        item,
        expected_model_version=settings.model_version,
        low_upper=float(settings.inference_risk_tier_low_upper),
        high_lower=float(settings.inference_risk_tier_high_lower),
    )


def test_validate_sync_metadata_rejects_risk_mismatch() -> None:
    item = _valid_item(risk="high")
    with pytest.raises(ClientHttpError) as exc:
        validate_sync_metadata_item(
            item,
            expected_model_version=settings.model_version,
            low_upper=float(settings.inference_risk_tier_low_upper),
            high_lower=float(settings.inference_risk_tier_high_lower),
        )
    assert exc.value.code == "sync_risk_mismatch"


def test_validate_sync_metadata_rejects_model_version() -> None:
    item = _valid_item(model_version="wrong")
    with pytest.raises(ClientHttpError) as exc:
        validate_sync_metadata_item(
            item,
            expected_model_version=settings.model_version,
            low_upper=float(settings.inference_risk_tier_low_upper),
            high_lower=float(settings.inference_risk_tier_high_lower),
        )
    assert exc.value.code == "sync_model_version_mismatch"
