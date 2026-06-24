#!/usr/bin/env python3
"""
Evaluación en un directorio test (p. ej. Ghana pediátrico) con calibración fija del ensemble v2.

Aplica ``T`` y τ operacional desde ``calibration_ensemble_ghana_v2.json``.

Uso::

    cd ml
    python scripts/evaluate_dir.py \\
        --test-dir data/ghana/test \\
        --calibration-json artifacts/runs/calibration_ensemble_ghana_v2.json \\
        --dataset-label ghana_ensemble_v2
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
from baseline.config import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    HEAD_LEARNING_RATE,
    MODEL_DIR,
    RUNS_DIR,
)
from baseline.dataops import load_test_dataset, write_json  # noqa: E402
from baseline.evaluation import (  # noqa: E402
    build_threshold_evaluation_results,
    collect_binary_predictions,
)
from baseline.model import compile_for_binary  # noqa: E402


def _load_calibration(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cal = data.get("calibration") or {}
    T = float(cal.get("temperature_T", 1.0))
    sel = cal.get("operational_threshold_selection") or {}
    tau = float(sel.get("threshold", 0.5))
    return T, tau


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluar modelo en test-dir externo con T/τ fijos.")
    p.add_argument("--model-path", type=Path, default=MODEL_DIR / DEFAULT_MODEL_NAME)
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument(
        "--calibration-json",
        type=Path,
        required=True,
        help="JSON de calibrate_eval (T y τ desde validación Nature).",
    )
    p.add_argument(
        "--dataset-label",
        type=str,
        default="external",
        help="Etiqueta en el informe (p. ej. ghana_external).",
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--ece-bins", type=int, default=15)
    return p.parse_args()


def main() -> None:
    from tensorflow import keras

    args = _parse_args()
    model_path = args.model_path.expanduser().resolve()
    test_dir = args.test_dir.expanduser().resolve()
    cal_path = args.calibration_json.expanduser().resolve()

    if not model_path.is_file():
        raise SystemExit(f"No existe --model-path: {model_path}")
    if not test_dir.is_dir():
        raise SystemExit(f"No existe --test-dir: {test_dir}")
    if not cal_path.is_file():
        raise SystemExit(f"No existe --calibration-json: {cal_path}")

    T, tau = _load_calibration(cal_path)
    model = keras.models.load_model(model_path, compile=False)
    compile_for_binary(model, HEAD_LEARNING_RATE)

    test_ds_eval = load_test_dataset(test_dir)
    raw_eval = model.evaluate(test_ds_eval, return_dict=True, verbose=1)
    loss_unc = float(raw_eval["loss"])
    auc_unc = float(raw_eval["auc"])

    test_ds_pred = load_test_dataset(test_dir)
    y_true, y_prob = collect_binary_predictions(model, test_ds_pred)
    y_cal = apply_temperature_scaling(y_prob, T)

    loss_cal = mean_binary_cross_entropy(y_true, y_cal)
    auc_cal = auc_roc_keras(y_true, y_cal)

    tau_source = f"fixed_from_calibration_json:{cal_path.name}"
    unc = enrich_binary_eval_with_calibration_metrics(
        build_threshold_evaluation_results(
            loss=loss_unc,
            auc_val=auc_unc,
            y_true=y_true,
            y_prob=y_prob,
            operational_threshold=tau,
            operational_threshold_source=tau_source,
        ),
        y_true,
        y_prob,
        n_ece_bins=args.ece_bins,
    )
    cali = enrich_binary_eval_with_calibration_metrics(
        build_threshold_evaluation_results(
            loss=loss_cal,
            auc_val=auc_cal,
            y_true=y_true,
            y_prob=y_cal,
            operational_threshold=tau,
            operational_threshold_source=tau_source,
        ),
        y_true,
        y_cal,
        n_ece_bins=args.ece_bins,
    )

    started = datetime.now(timezone.utc)
    run_id = f"eval_{args.dataset_label}_{started.strftime('%Y%m%dT%H%M%SZ')}"
    out_path = args.output_json or (RUNS_DIR / f"{run_id}.json")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "timestamp_utc": started.isoformat(),
        "dataset_label": args.dataset_label,
        "model_path": str(model_path),
        "test_dir": str(test_dir),
        "calibration_json": str(cal_path),
        "temperature_T_applied": T,
        "operational_threshold_applied": tau,
        "test_uncalibrated": unc,
        "test_calibrated": cali,
    }
    write_json(out_path, payload)

    mop = cali.get("at_operational_threshold") or {}
    print(f"Informe: {out_path}")
    print(f"  AUC (calibrado): {cali.get('auc', 0):.4f}")
    print(
        "  @τ fijo — precision: {:.4f}, recall: {:.4f}".format(
            float(mop.get("precision", 0)),
            float(mop.get("recall", 0)),
        ),
    )


if __name__ == "__main__":
    main()
