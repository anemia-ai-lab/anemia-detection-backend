#!/usr/bin/env python3
"""
Evalúa un ensemble (promedio raw) en test, con calibración opcional y métricas original-only.

Uso::

    python scripts/evaluate_ensemble.py \\
        --model-path artifacts/models/baseline_mobilenetv2_ghana_augmented_seed42.keras \\
        --model-path artifacts/models/baseline_mobilenetv2_ghana_augmented_seed123.keras \\
        --test-dir data/ghana/test \\
        --calibration-json artifacts/runs/calibration_ensemble_*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from baseline.calibration import (  # noqa: E402
    apply_temperature_scaling,
    auc_roc_keras,
    enrich_binary_eval_with_calibration_metrics,
    mean_binary_cross_entropy,
)
from baseline.config import DEFAULT_TEST_DIR, HEAD_LEARNING_RATE, RUNS_DIR  # noqa: E402
from baseline.dataops import load_test_dataset, write_json  # noqa: E402
from baseline.ensemble import ensemble_raw_probabilities, load_ensemble_models  # noqa: E402
from baseline.evaluation import build_threshold_evaluation_results  # noqa: E402
from baseline.model import compile_for_binary  # noqa: E402
from baseline.risk_tiers import risk_tier_from_probability  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluar ensemble en test.")
    p.add_argument("--model-path", type=Path, action="append", required=True)
    p.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument("--calibration-json", type=Path, default=None)
    p.add_argument("--dataset-label", type=str, default="ghana_test")
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def _tier_counts(
    y_true,
    p_cal,
    *,
    low_upper: float,
    high_lower: float,
) -> dict[str, int]:
    counts = {"low": 0, "medium": 0, "high": 0}
    for p in p_cal:
        counts[
            risk_tier_from_probability(float(p), low_upper=low_upper, high_lower=high_lower)
        ] += 1
    return counts


def main() -> None:
    args = _parse_args()
    test_dir = args.test_dir.expanduser().resolve()
    model_paths = [p.expanduser().resolve() for p in args.model_path]
    models = load_ensemble_models(model_paths)
    for m in models:
        compile_for_binary(m, HEAD_LEARNING_RATE)

    test_ds = load_test_dataset(test_dir)
    y_test, p_raw = ensemble_raw_probabilities(models, test_ds)

    T = 1.0
    tau = 0.5
    low_upper = 0.0
    high_lower = tau
    if args.calibration_json:
        cal_data = json.loads(args.calibration_json.read_text(encoding="utf-8"))
        cal_block = cal_data.get("calibration") or {}
        T = float(cal_block.get("temperature_T", 1.0))
        tiers = cal_block.get("risk_tier_thresholds") or {}
        low_upper = float(tiers.get("low_upper", 0.0))
        high_lower = float(tiers.get("high_lower", 0.5))

    p_cal = apply_temperature_scaling(p_raw, T)
    loss = mean_binary_cross_entropy(y_test, p_cal)
    auc = auc_roc_keras(y_test, p_cal)
    results = build_threshold_evaluation_results(
        loss=loss,
        auc_val=auc,
        y_true=y_test,
        y_prob=p_cal,
        operational_threshold=high_lower,
        operational_threshold_source="from_calibration_json",
    )
    results = enrich_binary_eval_with_calibration_metrics(results, y_test, p_cal)

    started = datetime.now(timezone.utc)
    run_id = f"eval_ensemble_{args.dataset_label}_{started.strftime('%Y%m%dT%H%M%SZ')}"
    payload = {
        "run_id": run_id,
        "timestamp_utc": started.isoformat(),
        "dataset_label": args.dataset_label,
        "test_dir": str(test_dir),
        "ensemble_member_paths": [str(p) for p in model_paths],
        "temperature_T": T,
        "risk_tier_thresholds": {"low_upper": low_upper, "high_lower": high_lower},
        "tier_sample_counts": _tier_counts(
            y_test,
            p_cal,
            low_upper=low_upper,
            high_lower=high_lower,
        ),
        "results": results,
    }
    out = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else (RUNS_DIR / f"{run_id}.json")
    )
    write_json(out, payload)
    print(f"AUC={auc:.4f} loss={loss:.4f}")
    print(f"JSON: {out}")


if __name__ == "__main__":
    main()
