#!/usr/bin/env python3
"""
CLI: inferencia offline TensorFlow Lite (G8).

Ejemplo::

    cd ml && PYTHONPATH=.. .venv/bin/python scripts/run_tflite_inference.py \\
        --image ../path/to/a.png \\
        --tflite-path artifacts/models/baseline_mobilenetv2_v1.tflite \\
        --metadata-path artifacts/models/baseline_mobilenetv2_v1.metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(ML_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inferencia TFLite offline con salida JSON.")
    p.add_argument(
        "--image",
        type=Path,
        action="append",
        default=[],
        help="Ruta de imagen (repetible).",
    )
    p.add_argument("--input-dir", type=Path, default=None, help="Directorio con imágenes.")
    p.add_argument(
        "--tflite-path",
        type=Path,
        default=ML_ROOT / "artifacts" / "models" / "baseline_mobilenetv2_v1.tflite",
    )
    p.add_argument(
        "--metadata-path",
        type=Path,
        default=ML_ROOT / "artifacts" / "models" / "baseline_mobilenetv2_v1.metadata.json",
    )
    p.add_argument("--output-json", type=Path, default=None, help="Escribir JSON además de stdout.")
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help=(
            "Opcional: escribe ml/artifacts/tflite_runs/<run_id>/outputs.json y manifest.json."
        ),
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identificador de corrida (default: UUID).",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _collect_image_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for im in args.image:
        paths.append(im.resolve())
    if args.input_dir is not None:
        d = args.input_dir.resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"--input-dir no es directorio: {d}")
        for child in sorted(d.iterdir()):
            if child.is_file() and child.suffix.lower() in _IMAGE_SUFFIXES:
                paths.append(child.resolve())
    # únicos, orden estables
    return sorted({p for p in paths if p.is_file()})


def main() -> int:
    import numpy as np
    import tensorflow as tf

    from ml.inference.tflite_inference import get_tflite_engine

    args = _parse_args()
    tf.keras.utils.set_random_seed(int(args.seed))
    np.random.seed(int(args.seed))

    try:
        paths = _collect_image_paths(args)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    if not paths:
        print("No hay imágenes: use --image y/o --input-dir.", file=sys.stderr)
        return 1

    run_id = args.run_id or str(uuid.uuid4())
    tflite = args.tflite_path.resolve()
    meta = args.metadata_path.resolve()

    engine = get_tflite_engine(str(tflite), str(meta))
    images = [p.read_bytes() for p in paths]
    results = engine.predict_batch(images)

    out_results = []
    for p, r in zip(paths, results, strict=True):
        d = r.to_json_dict()
        d["source_path"] = str(p)
        out_results.append(d)

    payload = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "tflite_path": str(tflite),
        "metadata_path": str(meta),
        "seed": int(args.seed),
        "results": out_results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)

    if args.output_json is not None:
        out_path = args.output_json.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            print(f"Refusing to overwrite {out_path} (use --overwrite)", file=sys.stderr)
            return 2
        out_path.write_text(text, encoding="utf-8")

    if args.artifacts_dir is not None:
        base = (args.artifacts_dir / run_id).resolve()
        if base.exists() and not args.overwrite:
            print(f"Refusing to write under existing {base} (use --overwrite)", file=sys.stderr)
            return 2
        base.mkdir(parents=True, exist_ok=True)
        (base / "outputs.json").write_text(text, encoding="utf-8")
        manifest = {
            "run_id": run_id,
            "tflite_path": str(tflite),
            "metadata_path": str(meta),
            "image_paths": [str(p) for p in paths],
            "seed": int(args.seed),
        }
        (base / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
