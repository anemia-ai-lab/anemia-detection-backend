#!/usr/bin/env python3
"""
Lee un informe ``calibration_*.json`` y emite T / τ operacional para backend y export TFLite.

Uso::

    python ml/scripts/sync_calibration_constants.py \\
        --calibration-json ml/artifacts/runs/calibration_20260420T045056Z.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sincronizar T y τ desde informe de calibración.")
    p.add_argument(
        "--calibration-json",
        type=Path,
        required=True,
        help="JSON generado por calibrate_eval.py",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    path = args.calibration_json.resolve()
    if not path.is_file():
        print(f"No existe: {path}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    cal = data.get("calibration") or {}
    T = float(cal.get("temperature_T", 0))
    if T <= 0:
        print("temperature_T inválido o ausente en calibration", file=sys.stderr)
        return 1
    sel = cal.get("operational_threshold_selection") or {}
    thr_block = (data.get("test_calibrated") or {}).get("thresholds_used") or {}
    tau = float(sel.get("threshold") or thr_block.get("operational_threshold") or 0)
    tiers = cal.get("risk_tier_thresholds") or {}
    low_upper = float(tiers.get("low_upper", 0))
    high_lower = float(tiers.get("high_lower", tau))
    print(f"INFERENCE_CALIBRATION_TEMPERATURE={T}")
    print(f"INFERENCE_CALIBRATION_OPERATIONAL_THRESHOLD={high_lower}")
    print(f"INFERENCE_RISK_TIER_LOW_UPPER={low_upper}")
    print(f"INFERENCE_RISK_TIER_HIGH_LOWER={high_lower}")
    print()
    print("# export_tflite.py")
    print(f"TEMPERATURE = {T}")
    print(f"OPERATIONAL_THRESHOLD = {high_lower}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
