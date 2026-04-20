#!/usr/bin/env python3
"""
Evalúa un modelo .keras guardado sobre una carpeta ``test/`` con dos clases.

Mismas convenciones que ``train.py`` en test: AUC, umbral teórico 0.5 (referencia) y
**umbral operacional** (ROC-Youden) como métricas principales para informes clínicos.

Uso::

    cd ml && pip install -r requirements.txt
    python scripts/evaluate.py --model-path artifacts/models/baseline_mobilenetv2.keras \\
        --test-dir data/test

Calibración post-hoc (*temperature scaling*, sin reentrenar): ver ``scripts/calibrate_eval.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from baseline.config import (  # noqa: E402
    DEFAULT_METRICS_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEST_DIR,
    METRICS_DIR,
    MODEL_DIR,
)
from baseline.dataops import load_test_dataset, write_json  # noqa: E402
from baseline.evaluation import (  # noqa: E402
    build_threshold_evaluation_results,
    collect_binary_predictions,
)
from tensorflow import keras  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluar baseline MobileNetV2 en carpeta test/.")
    p.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_DIR / DEFAULT_MODEL_NAME,
        help="Archivo .keras entrenado.",
    )
    p.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="Carpeta con dos subcarpetas de clase (misma convención que train/).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=METRICS_DIR / DEFAULT_METRICS_NAME,
        help="Ruta del JSON con evaluación (incluye umbral operacional y referencia 0.5).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.model_path.is_file():
        raise SystemExit(f"No existe el modelo: {args.model_path}")
    if not args.test_dir.is_dir():
        raise SystemExit(f"No existe --test-dir: {args.test_dir}")

    model = keras.models.load_model(args.model_path)
    test_ds_eval = load_test_dataset(args.test_dir)
    raw_eval = model.evaluate(test_ds_eval, return_dict=True, verbose=1)
    loss_v = float(raw_eval["loss"])
    auc_v = float(raw_eval["auc"])

    test_ds_pred = load_test_dataset(args.test_dir)
    y_true, y_prob = collect_binary_predictions(model, test_ds_pred)
    out = build_threshold_evaluation_results(loss=loss_v, auc_val=auc_v, y_true=y_true, y_prob=y_prob)

    write_json(args.output_json, out)
    mop = out.get("at_operational_threshold") or {}
    print(f"Métricas guardadas en: {args.output_json}")
    print(
        "  [operacional] τ={:.6f} — precision: {:.6f}, recall: {:.6f}".format(
            float(mop.get("threshold", 0.0)),
            float(mop.get("precision", 0.0)),
            float(mop.get("recall", 0.0)),
        ),
    )
    m5 = out.get("at_threshold_0_5") or {}
    print(
        "  [referencia 0.5] precision: {:.6f}, recall: {:.6f}".format(
            float(m5.get("precision", 0.0)),
            float(m5.get("recall", 0.0)),
        ),
    )


if __name__ == "__main__":
    main()
