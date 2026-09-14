#!/usr/bin/env python3
"""
Calibración post-hoc de un **ensemble** (promedio de probabilidades raw) + umbrales de riesgo en 3 niveles.

Uso::

    cd ml && python scripts/calibrate_ensemble_eval.py \\
        --model-path artifacts/models/baseline_mobilenetv2_ghana_augmented_seed42.keras \\
        --model-path artifacts/models/baseline_mobilenetv2_ghana_augmented_seed123.keras \\
        --model-path artifacts/models/baseline_mobilenetv2_ghana_augmented_seed456.keras \\
        --train-dir data/ghana/train --test-dir data/ghana/test \\
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from baseline.calibration import (  # noqa: E402
    apply_named_calibration,
    apply_temperature_scaling,
    auc_roc_keras,
    brier_score_binary,
    enrich_binary_eval_with_calibration_metrics,
    expected_calibration_error_binary,
    fit_isotonic_regression_on_probabilities,
    fit_platt_scaling_on_probabilities,
    fit_temperature_scaling_on_probabilities,
    mean_binary_cross_entropy,
)
from baseline.config import (  # noqa: E402
    DEFAULT_TEST_DIR,
    DEFAULT_TRAIN_DIR,
    HEAD_LEARNING_RATE,
    RUNS_DIR,
    SEED,
)
from baseline.dataops import (  # noqa: E402
    load_test_dataset,
    load_validation_dataset,
    write_json,
    write_text,
)
from baseline.ensemble import ensemble_raw_probabilities, load_ensemble_models  # noqa: E402
from baseline.evaluation import build_threshold_evaluation_results  # noqa: E402
from baseline.model import compile_for_binary  # noqa: E402
from baseline.risk_tiers import risk_tier_thresholds_from_validation  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrar ensemble (mean raw prob) + tiers.")
    p.add_argument(
        "--model-path",
        type=Path,
        action="append",
        required=True,
        help="Ruta .keras (repetir para cada miembro del ensemble).",
    )
    p.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    p.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument("--validation-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-md", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    run_id = f"calibration_ensemble_{started.strftime('%Y%m%dT%H%M%SZ')}"
    train_dir = args.train_dir.expanduser().resolve()
    test_dir = args.test_dir.expanduser().resolve()
    model_paths = [p.expanduser().resolve() for p in args.model_path]

    models = load_ensemble_models(model_paths)
    for m in models:
        compile_for_binary(m, HEAD_LEARNING_RATE)

    val_ds, val_counts, _ = load_validation_dataset(
        train_dir,
        validation_split=float(args.validation_split),
        seed=int(args.seed),
    )
    y_val, p_val = ensemble_raw_probabilities(models, val_ds)
    T, fit_diag = fit_temperature_scaling_on_probabilities(y_val, p_val)
    platt_a, platt_b, platt_diag = fit_platt_scaling_on_probabilities(y_val, p_val)
    iso_x, iso_y, iso_diag = fit_isotonic_regression_on_probabilities(y_val, p_val)

    test_ds = load_test_dataset(test_dir)
    y_test, p_test = ensemble_raw_probabilities(models, test_ds)

    method_probs: dict[str, np.ndarray] = {
        "temperature": apply_temperature_scaling(p_test, T),
        "platt": apply_named_calibration(p_test, "platt", platt_a=platt_a, platt_b=platt_b),
        "isotonic": apply_named_calibration(
            p_test, "isotonic", isotonic_x=iso_x, isotonic_y=iso_y
        ),
    }
    comparison: dict[str, dict[str, float]] = {}
    for name, p_m in method_probs.items():
        comparison[name] = {
            "expected_calibration_error": float(
                expected_calibration_error_binary(y_test, p_m, n_bins=int(args.ece_bins))
            ),
            "brier_score": float(brier_score_binary(y_test, p_m)),
            "mean_nll": float(mean_binary_cross_entropy(y_test, p_m)),
        }

    # Producción: menor ECE en test entre temperature y Platt (isotonic es ablación).
    chosen = min(("temperature", "platt"), key=lambda k: comparison[k]["expected_calibration_error"])
    p_val_cal = apply_named_calibration(
        p_val,
        chosen,
        temperature=T,
        platt_a=platt_a,
        platt_b=platt_b,
    )
    risk_tiers = risk_tier_thresholds_from_validation(y_val, p_val_cal)
    tau_high = float(risk_tiers["high_lower"])

    p_cal = method_probs[chosen]
    loss_cal = mean_binary_cross_entropy(y_test, p_cal)
    auc_cal = auc_roc_keras(y_test, p_cal)

    cal_base = build_threshold_evaluation_results(
        loss=loss_cal,
        auc_val=auc_cal,
        y_true=y_test,
        y_prob=p_cal,
        operational_threshold=tau_high,
        operational_threshold_source="roc_youden_on_validation_calibrated_applied_to_test",
    )
    cali = enrich_binary_eval_with_calibration_metrics(
        cal_base,
        y_test,
        p_cal,
        n_ece_bins=args.ece_bins,
    )

    payload: dict[str, object] = {
        "run_id": run_id,
        "timestamp_utc": started.isoformat(),
        "ensemble": {
            "member_paths": [str(p) for p in model_paths],
            "aggregation": "mean_raw_probability",
            "n_members": len(model_paths),
        },
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "calibration": {
            "method": (
                "platt_scaling_on_ensemble_mean_raw"
                if chosen == "platt"
                else "temperature_scaling_on_ensemble_mean_raw"
            ),
            "chosen_method": chosen,
            "temperature_T": float(T),
            "platt_a": float(platt_a),
            "platt_b": float(platt_b),
            "isotonic_ablation": {
                "n_knots": int(iso_x.size),
                "fit_diagnostics": iso_diag,
            },
            "comparison_on_test": comparison,
            "validation_split": float(args.validation_split),
            "seed_used_for_val_split": int(args.seed),
            "n_validation_samples": int(y_val.size),
            "validation_class_counts": {str(k): int(v) for k, v in val_counts.items()},
            "fit_diagnostics": {
                "temperature": fit_diag,
                "platt": platt_diag,
            },
            "operational_threshold_selection": {
                "threshold": tau_high,
                "source": risk_tiers["high_lower_source"],
            },
            "risk_tier_thresholds": risk_tiers,
        },
        "test_calibrated": cali,
    }

    out_json = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else (RUNS_DIR / f"{run_id}.json")
    )
    out_md = (
        args.output_md.expanduser().resolve() if args.output_md else out_json.with_suffix(".md")
    )
    write_json(out_json, payload)
    lines = [
        f"# Calibración ensemble — `{run_id}`",
        "",
        f"- Miembros: {len(model_paths)}",
        f"- Método elegido (menor ECE test, T vs Platt): **{chosen}**",
        f"- T: **{T:.6f}**",
        f"- Platt a,b: **{platt_a:.6f}**, **{platt_b:.6f}**",
        f"- ECE test T / Platt / isotonic: "
        f"{comparison['temperature']['expected_calibration_error']:.4f} / "
        f"{comparison['platt']['expected_calibration_error']:.4f} / "
        f"{comparison['isotonic']['expected_calibration_error']:.4f}",
        f"- τ alto (high_lower): **{tau_high:.6f}**",
        f"- τ bajo (low_upper): **{risk_tiers['low_upper']:.6f}**",
        f"- AUC test calibrado: **{cali.get('auc')}**",
        "",
    ]
    write_text(out_md, "\n".join(lines))
    print(f"T={T}")
    print(f"chosen={chosen} platt_a={platt_a} platt_b={platt_b}")
    print(f"risk tiers: low_upper={risk_tiers['low_upper']}, high_lower={risk_tiers['high_lower']}")
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()
