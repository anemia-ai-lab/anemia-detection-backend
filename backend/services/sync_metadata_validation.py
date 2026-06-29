"""Coherencia interna de metadatos sync offline (sin re-inferencia)."""

from __future__ import annotations

from backend.core.exceptions import ClientHttpError
from backend.core.prometheus_metrics import record_sync_metadata_rejected
from backend.core.risk_mapping import risk_from_probability
from backend.inference.probability_calibration import binary_prediction_from_threshold
from backend.schemas.prediction import PredictionSyncMetadataItem

_SCORE_TOLERANCE = 1e-4


def validate_sync_metadata_item(
    item: PredictionSyncMetadataItem,
    *,
    expected_model_version: str,
    low_upper: float,
    high_lower: float,
) -> None:
    """Rechaza payloads incoherentes antes de insertar (422)."""
    if item.model_version != expected_model_version:
        record_sync_metadata_rejected("model_version_mismatch")
        raise ClientHttpError(
            f"model_version debe ser {expected_model_version!r}.",
            422,
            code="sync_model_version_mismatch",
        )

    if abs(item.score - item.calibrated_probability) > _SCORE_TOLERANCE:
        record_sync_metadata_rejected("score_mismatch")
        raise ClientHttpError(
            "score debe coincidir con calibrated_probability.",
            422,
            code="sync_score_mismatch",
        )

    expected_risk = risk_from_probability(
        item.calibrated_probability,
        low_upper=low_upper,
        high_lower=high_lower,
    )
    if item.risk != expected_risk:
        record_sync_metadata_rejected("risk_mismatch")
        raise ClientHttpError(
            "risk no coincide con calibrated_probability y los umbrales de tier.",
            422,
            code="sync_risk_mismatch",
        )

    expected_prediction = binary_prediction_from_threshold(
        item.calibrated_probability,
        item.threshold_used,
    )
    if item.prediction != expected_prediction:
        record_sync_metadata_rejected("prediction_mismatch")
        raise ClientHttpError(
            "prediction no coincide con calibrated_probability y threshold_used.",
            422,
            code="sync_prediction_mismatch",
        )
