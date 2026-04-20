#!/usr/bin/env python3
"""
Evalúa un modelo .keras guardado sobre una carpeta ``test/`` con dos clases.

Uso::

    cd ml && pip install -r requirements.txt
    python scripts/evaluate.py --model-path artifacts/models/baseline_mobilenetv2.keras \\
        --test-dir data/test
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
        help="Ruta del JSON con loss y métricas.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.model_path.is_file():
        raise SystemExit(f"No existe el modelo: {args.model_path}")
    if not args.test_dir.is_dir():
        raise SystemExit(f"No existe --test-dir: {args.test_dir}")

    model = keras.models.load_model(args.model_path)
    test_ds = load_test_dataset(args.test_dir)
    results = model.evaluate(test_ds, return_dict=True, verbose=1)
    out = {k: float(v) for k, v in results.items()}
    write_json(args.output_json, out)
    print(f"Métricas guardadas en: {args.output_json}")


if __name__ == "__main__":
    main()
