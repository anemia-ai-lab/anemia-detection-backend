"""Tests del parser y split de prepare_ghana_dataset (sin TensorFlow)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "ml" / "scripts" / "prepare_ghana_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_ghana_dataset", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prepare_ghana_dataset"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ghana():
    return _load_module()


def test_parse_anemic_fn(ghana) -> None:
    assert ghana.parse_ghana_filename("Anemic-FN-117.png") == (
        "positive",
        "ghana_fn_positive_117",
        "ghana_fn",
    )


def test_parse_non_anrmic_augmented_stem(ghana) -> None:
    assert ghana.parse_ghana_filename("Non-Anrmic-FN-188 (8).png") == (
        "negative",
        "ghana_fn_negative_188",
        "ghana_fn",
    )


def test_parse_anmeic_typo(ghana) -> None:
    assert ghana.parse_ghana_filename("Anmeic-fn-004 1 5.png") == (
        "positive",
        "ghana_anmeic_positive_4",
        "ghana_anmeic",
    )


def test_canonical_stem(ghana) -> None:
    assert ghana.is_canonical_original_stem("Anemic-FN-117")
    assert not ghana.is_canonical_original_stem("Anemic-FN-117 (2)")


def test_pick_canonical_prefers_clean_name(ghana, tmp_path: Path) -> None:
    a = tmp_path / "Anemic-FN-004.png"
    b = tmp_path / "Anemic-FN-004 (2).png"
    a.write_bytes(b"x" * 8)
    b.write_bytes(b"x" * 8)
    picked = ghana.pick_canonical_file([b, a])
    assert picked.name == "Anemic-FN-004.png"


def test_split_subjects_reproducible(ghana) -> None:
    keys = [f"ghana_fn_positive_{i}" for i in range(20)]
    a = ghana.split_subjects(keys, test_size=0.2, seed=42)
    b = ghana.split_subjects(keys, test_size=0.2, seed=42)
    assert a == b
    assert sum(1 for v in a.values() if v == "test") >= 1
