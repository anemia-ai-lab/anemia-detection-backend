#!/usr/bin/env python3
"""Exporta cada miembro del ensemble a .tflite y un metadata.json compartido."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from baseline.config import MODEL_DIR  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exportar ensemble a TFLite + metadata.")
    p.add_argument("--keras-path", type=Path, action="append", required=True)
    p.add_argument("--calibration-json", type=Path, required=True)
    p.add_argument(
        "--output-metadata",
        type=Path,
        default=MODEL_DIR / "baseline_mobilenetv2_ghana_ensemble.metadata.json",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _load_calibration(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    cal = data.get("calibration") or {}
    T = float(cal.get("temperature_T", 0))
    tiers = cal.get("risk_tier_thresholds") or {}
    sel = cal.get("operational_threshold_selection") or {}
    tau = float(sel.get("threshold") or tiers.get("high_lower") or 0)
    if T <= 0 or not 0 <= tau <= 1:
        raise ValueError(f"Calibración inválida en {path}")
    return {
        "temperature": T,
        "operational_threshold": tau,
        "risk_tier_thresholds": {
            "low_upper": float(tiers.get("low_upper", 0)),
            "high_lower": float(tiers.get("high_lower", tau)),
        },
    }


def main() -> int:
    args = _parse_args()
    cal = _load_calibration(args.calibration_json.resolve())
    export_script = _ML_ROOT / "scripts" / "export_tflite.py"
    members: list[str] = []

    for keras_path in args.keras_path:
        kp = keras_path.expanduser().resolve()
        stem = kp.stem
        out_tflite = MODEL_DIR / f"{stem}.tflite"
        out_meta = MODEL_DIR / f"{stem}.metadata.json"
        cmd = [
            sys.executable,
            str(export_script),
            "--keras-path",
            str(kp),
            "--output-tflite",
            str(out_tflite),
            "--output-metadata",
            str(out_meta),
            "--calibration-json",
            str(args.calibration_json.resolve()),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        rc = subprocess.run(cmd, cwd=_ML_ROOT, check=False)
        if rc.returncode != 0:
            return rc.returncode
        members.append(out_tflite.name)

    meta_path = args.output_metadata.expanduser().resolve()
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_meta = {
        "model_version": "v2.0-ensemble",
        "ensemble_members": members,
        "ensemble_aggregation": "mean_raw_probability",
        "per_hand_nail_aggregation": "max_calibrated_probability",
        "temperature": cal["temperature"],
        "operational_threshold": cal["operational_threshold"],
        "risk_tier_thresholds": cal["risk_tier_thresholds"],
        "preprocessing": "mobilenet_v2.preprocess_input",
        "calibration_required": True,
        "notes": (
            "Por uña: promediar raw_prob de los 3 TFLite, aplicar T, luego max entre anular/medio/índice "
            "para tiers bajo/medio/alto."
        ),
    }
    meta_path.write_text(json.dumps(ensemble_meta, indent=2), encoding="utf-8")
    print(f"Metadata ensemble: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
