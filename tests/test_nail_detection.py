"""Tests unitarios de recorte multinail (sin MediaPipe real)."""

from dataclasses import dataclass

import numpy as np
import pytest

from backend.inference.nail_detection import (
    EmptyNailDetector,
    FixedNailDetector,
    crop_from_normalized_roi,
    fallback_crops,
    landmarks_to_crops,
    normalize_rois_input,
    parse_roi_overrides,
)
from backend.services.exceptions import PredictionServiceError


@dataclass
class _FakeLandmark:
    x: float
    y: float
    visibility: float | None = 1.0
    presence: float | None = None


def _synthetic_hand_landmarks() -> list[_FakeLandmark]:
    """21 landmarks mínimos con tips/DIPs en posiciones utilizables."""
    landmarks = [_FakeLandmark(x=0.5, y=0.5) for _ in range(21)]
    landmarks[8] = _FakeLandmark(x=0.7, y=0.3)  # index tip
    landmarks[7] = _FakeLandmark(x=0.65, y=0.45)  # index DIP
    landmarks[12] = _FakeLandmark(x=0.5, y=0.28)  # middle tip
    landmarks[11] = _FakeLandmark(x=0.5, y=0.42)  # middle DIP
    landmarks[16] = _FakeLandmark(x=0.3, y=0.32)  # ring tip
    landmarks[15] = _FakeLandmark(x=0.35, y=0.46)  # ring DIP
    return landmarks


def test_fallback_vertical_thirds_returns_three_crops() -> None:
    rgb = np.full((90, 120, 3), 200, dtype=np.uint8)
    crops = fallback_crops(rgb, "vertical_thirds")
    assert len(crops) == 3
    assert [c.finger for c in crops] == ["index", "middle", "ring"]


def test_fallback_whole_returns_single_hand_crop() -> None:
    rgb = np.full((64, 64, 3), 180, dtype=np.uint8)
    crops = fallback_crops(rgb, "whole")
    assert len(crops) == 1
    assert crops[0].finger == "hand"
    assert crops[0].rgb.shape == rgb.shape


def test_crop_from_normalized_roi() -> None:
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = crop_from_normalized_roi(rgb, finger="index", x=0.1, y=0.2, w=0.3, h=0.4)
    assert crop is not None
    assert crop.finger == "index"
    assert crop.rgb.shape[0] == 40
    assert crop.rgb.shape[1] == 60


def test_normalize_rois_input_treats_empty_and_swagger_placeholder_as_none() -> None:
    assert normalize_rois_input(None) is None
    assert normalize_rois_input("") is None
    assert normalize_rois_input("   ") is None
    assert normalize_rois_input("string") is None
    assert normalize_rois_input("  STRING  ") is None
    assert normalize_rois_input("null") is None
    assert normalize_rois_input('[{"x":0.1,"y":0.1,"w":0.2,"h":0.2}]') is not None


def test_parse_roi_overrides_empty_list_returns_none() -> None:
    assert parse_roi_overrides("[]") is None


def test_parse_roi_overrides_valid_list() -> None:
    data = parse_roi_overrides('[{"finger":"index","x":0.1,"y":0.1,"w":0.2,"h":0.2}]')
    assert data is not None
    assert len(data) == 1
    assert data[0]["finger"] == "index"


def test_parse_roi_overrides_invalid_json() -> None:
    with pytest.raises(PredictionServiceError) as exc:
        parse_roi_overrides("{not-json")
    assert exc.value.code == "invalid_rois_json"


def test_landmarks_to_crops_synthetic_three_fingers() -> None:
    rgb = np.full((100, 100, 3), 180, dtype=np.uint8)
    crops = landmarks_to_crops(
        rgb,
        _synthetic_hand_landmarks(),
        min_confidence=0.5,
        crop_scale=1.0,
    )
    assert len(crops) == 3
    assert [c.finger for c in crops] == ["index", "middle", "ring"]


def test_fixed_nail_detector_three_crops() -> None:
    rgb = np.full((60, 90, 3), 150, dtype=np.uint8)
    crops = FixedNailDetector(count=3).detect(rgb)
    assert len(crops) == 3


def test_empty_nail_detector() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    assert EmptyNailDetector().detect(rgb) == []
