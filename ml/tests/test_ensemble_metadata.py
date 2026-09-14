"""Contrato móvil v2: metadata del ensemble 3× (sin cargar los tres TFLite)."""

from __future__ import annotations

import json

import pytest

from ml.tests.conftest import ENSEMBLE_METADATA, ML_ROOT


def test_ensemble_metadata_lists_three_members() -> None:
    if not ENSEMBLE_METADATA.is_file():
        pytest.skip(f"Metadata ensemble no encontrada: {ENSEMBLE_METADATA}")
    data = json.loads(ENSEMBLE_METADATA.read_text(encoding="utf-8"))
    assert data["model_version"] == "v2.0-ensemble"
    members = data["ensemble_members"]
    assert len(members) == 3
    for name in members:
        assert (ML_ROOT / "artifacts" / "models" / name).is_file(), f"Falta miembro: {name}"
    assert data["ensemble_aggregation"] == "mean_raw_probability"
    assert data["per_hand_nail_aggregation"] == "median_calibrated_probability"
    assert data["nail_aggregation_even_n"] == "lower_central"
    assert int(data["min_nail_count"]) == 2
    assert float(data["crop_scale"]) == pytest.approx(1.0)
    assert data.get("crop") == "tip_to_dip_rotated"
    assert float(data["temperature"]) > 0
    tiers = data["risk_tier_thresholds"]
    assert float(tiers["low_upper"]) < float(tiers["high_lower"])
