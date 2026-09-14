"""Agregación por uña: median lower-even, max, majority_2_of_3."""

from __future__ import annotations

import statistics

import pytest

from backend.inference.nail_aggregation import select_aggregated_nail_index


def test_three_nails_median_is_central_not_max() -> None:
    cals = [0.2, 0.8, 0.3]
    fingers = ["index", "middle", "ring"]
    idx = select_aggregated_nail_index(cals, fingers=fingers, mode="median")
    assert fingers[idx] == "ring"
    assert cals[idx] == pytest.approx(0.3)
    idx_max = select_aggregated_nail_index(cals, fingers=fingers, mode="max")
    assert fingers[idx_max] == "middle"
    assert cals[idx_max] == pytest.approx(0.8)
    idx_maj = select_aggregated_nail_index(cals, fingers=fingers, mode="majority_2_of_3")
    assert idx_maj == idx


def test_two_nails_median_is_min_not_statistics_median() -> None:
    cals = [0.9, 0.2]
    fingers = ["index", "middle"]
    idx = select_aggregated_nail_index(cals, fingers=fingers, mode="median")
    assert cals[idx] == pytest.approx(0.2)
    assert cals[idx] != pytest.approx(statistics.median(cals))
    assert statistics.median(cals) == pytest.approx(0.55)
    idx_maj = select_aggregated_nail_index(cals, fingers=fingers, mode="majority_2_of_3")
    assert cals[idx_maj] == pytest.approx(0.2)
    idx_max = select_aggregated_nail_index(cals, fingers=fingers, mode="max")
    assert cals[idx_max] == pytest.approx(0.9)


def test_single_nail_all_modes_agree() -> None:
    cals = [0.77]
    fingers = ["index"]
    for mode in ("median", "max", "majority_2_of_3"):
        idx = select_aggregated_nail_index(cals, fingers=fingers, mode=mode)
        assert idx == 0


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        select_aggregated_nail_index([], fingers=[], mode="median")


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        select_aggregated_nail_index([0.1], fingers=["index"], mode="mean")
