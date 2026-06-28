"""Tests for Render URL guard in scripts/smoke_prod.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_smoke_prod():
    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_prod.py"
    spec = importlib.util.spec_from_file_location("smoke_prod", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_is_retired_render_url_blocks_onrender_host() -> None:
    smoke = _load_smoke_prod()
    assert smoke.is_retired_render_url("https://anemia-detection-backend.onrender.com")
    assert smoke.is_retired_render_url("http://app.onrender.com")


def test_is_retired_render_url_allows_alb() -> None:
    smoke = _load_smoke_prod()
    assert not smoke.is_retired_render_url(
        "http://Anemia-ApiSe-6KjU9Jqn3PHi-65155833.us-west-2.elb.amazonaws.com",
    )


def test_is_retired_render_url_no_false_positive_on_substring_in_path() -> None:
    smoke = _load_smoke_prod()
    assert not smoke.is_retired_render_url("http://evil.onrender.com.attacker.com")
