#!/usr/bin/env python3
"""
Entrena el baseline MobileNetV2 (transfer learning: cabezal + fine-tuning opcional).

Uso (desde la raíz del repo o desde ``ml/``)::

    cd ml && pip install -r requirements.txt
    python scripts/train.py --demo
    python scripts/train.py --train-dir data/train --fine-tune-epochs 2
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from baseline.config import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_TRAIN_DIR,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_FREEZE_UP_TO_LAYER,
    FINE_TUNE_LEARNING_RATE,
    HEAD_EPOCHS,
    HEAD_LEARNING_RATE,
    MODEL_DIR,
    RUNS_DIR,
)
from baseline.dataops import load_image_datasets, make_demo_datasets, write_json  # noqa: E402
from baseline.model import (  # noqa: E402
    build_model,
    compile_for_binary,
    set_backbone_trainable,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenar baseline MobileNetV2 (binario).")
    p.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Carpeta con dos subcarpetas de clase (p. ej. negative/ y positive/).",
    )
    p.add_argument(
        "--output-model",
        type=Path,
        default=None,
        help="Ruta del .keras (por defecto artifacts/models/baseline_mobilenetv2.keras).",
    )
    p.add_argument("--head-epochs", type=int, default=HEAD_EPOCHS)
    p.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    p.add_argument(
        "--demo",
        action="store_true",
        help="Ignora imágenes en disco y entrena con batches sintéticos (smoke test).",
    )
    p.add_argument("--validation-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = args.output_model or (MODEL_DIR / DEFAULT_MODEL_NAME)

    if args.demo:
        train_ds, val_ds = make_demo_datasets()
    else:
        if not args.train_dir.is_dir():
            raise SystemExit(f"No existe --train-dir: {args.train_dir}")
        train_ds, val_ds = load_image_datasets(
            args.train_dir,
            validation_split=args.validation_split,
            seed=args.seed,
        )

    model = build_model(backbone_trainable=False)
    compile_for_binary(model, HEAD_LEARNING_RATE)

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        verbose=1,
    )

    history_ft = None
    if args.fine_tune_epochs > 0:
        set_backbone_trainable(model, FINE_TUNE_FREEZE_UP_TO_LAYER)
        compile_for_binary(model, FINE_TUNE_LEARNING_RATE)
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            verbose=1,
        )

    model.save(out_path)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {
        "run_id": run_id,
        "model_path": str(out_path),
        "demo": args.demo,
        "train_dir": str(args.train_dir) if not args.demo else None,
        "head_epochs": args.head_epochs,
        "fine_tune_epochs": args.fine_tune_epochs,
        "head_history": {k: [float(x) for x in v] for k, v in history_head.history.items()},
        "fine_tune_history": (
            {k: [float(x) for x in v] for k, v in history_ft.history.items()}
            if history_ft
            else None
        ),
    }
    write_json(RUNS_DIR / f"train_{run_id}.json", meta)
    print(f"Modelo guardado en: {out_path}")
    print(f"Metadatos de run: {RUNS_DIR / f'train_{run_id}.json'}")


if __name__ == "__main__":
    main()
