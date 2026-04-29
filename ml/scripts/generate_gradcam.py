#!/usr/bin/env python3
"""
CLI: generación de artefactos Grad-CAM (G10).

Ejemplo::

    cd ml && PYTHONPATH=.. .venv/bin/python scripts/generate_gradcam.py \\
        --image path/to/a.png \\
        --output-dir artifacts/explainability
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(ML_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

logger = logging.getLogger(__name__)

DISCLAIMER_NON_DIAGNOSTIC = (
    "Non-diagnostic research visualization only. Not medical advice or a clinical decision."
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grad-CAM para el modelo baseline MobileNetV2.")
    p.add_argument(
        "--image",
        type=Path,
        action="append",
        default=[],
        help="Ruta de imagen (repetible).",
    )
    p.add_argument("--input-dir", type=Path, default=None, help="Directorio con imágenes.")
    p.add_argument(
        "--model-path",
        type=Path,
        default=ML_ROOT / "artifacts" / "models" / "baseline_mobilenetv2.keras",
    )
    p.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Nombre de capa convolucional (override).",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="ID de corrida (default: timestamp + uuid corto).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ML_ROOT / "artifacts" / "explainability",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _collect_images(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for im in args.image:
        paths.append(im.resolve())
    if args.input_dir is not None:
        d = args.input_dir.resolve()
        if not d.is_dir():
            msg = f"--input-dir no es directorio: {d}"
            raise FileNotFoundError(msg)
        for child in sorted(d.iterdir()):
            if child.is_file() and child.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                paths.append(child.resolve())
    return sorted({p for p in paths if p.is_file()})


def _save_uint8_png(arr: object, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG")


def _preprocessed_to_display_png(batch: object) -> object:
    """Visualización aproximada del tensor de entrada del modelo (batch 1, H, W, 3) float."""
    import numpy as np

    x = batch[0].astype(np.float64)
    lo = np.min(x)
    hi = np.max(x)
    if hi - lo < 1e-9:
        return np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
    y = (x - lo) / (hi - lo) * 255.0
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


def main() -> int:
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    from backend.core.config import settings
    from ml.explainability.gradcam import GradCAM
    from ml.preprocessing.pipeline import preprocess_image_bytes

    args = _parse_args()
    level = logging.DEBUG if (args.verbose or args.debug) else logging.INFO
    logging.basicConfig(level=level)
    tf.keras.utils.set_random_seed(int(args.seed))

    try:
        paths = _collect_images(args)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    if not paths:
        print("No hay imágenes.", file=sys.stderr)
        return 1

    model_path = args.model_path.resolve()
    if not model_path.is_file():
        print(f"No existe el modelo: {model_path}", file=sys.stderr)
        return 1

    run_id = args.run_id or f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    out_root = (args.output_dir / run_id).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(model_path, compile=False)
    gc = GradCAM(model, layer_name=args.layer)

    for img_path in paths:
        raw = img_path.read_bytes()
        pre = preprocess_image_bytes(raw)
        rgb = pre.decoded_rgb_uint8
        res, pred, risk, risk_label = gc.explain_with_decision(rgb)

        stem = img_path.stem
        dest = out_root / stem
        dest.mkdir(parents=True, exist_ok=True)

        _save_uint8_png(rgb, dest / "original.png")
        _save_uint8_png(_preprocessed_to_display_png(res.preprocessed), dest / "preprocessed.png")
        hm_u8 = np.clip(np.round(res.heatmap * 255.0), 0, 255).astype(np.uint8)
        _save_uint8_png(
            np.stack([hm_u8, hm_u8, hm_u8], axis=-1),
            dest / "heatmap.png",
        )
        _save_uint8_png(res.overlay, dest / "overlay.png")
        _save_uint8_png(
            np.stack([res.saliency, res.saliency, res.saliency], axis=-1),
            dest / "saliency.png",
        )

        meta = {
            "model_version": settings.model_version,
            "selected_layer": res.selected_layer,
            "raw_probability": res.raw_probability,
            "calibrated_probability": res.calibrated_probability,
            "threshold_used": float(settings.inference_calibration_operational_threshold),
            "prediction": int(pred),
            "risk": risk,
            "risk_label": risk_label,
            "inference_mode": "keras_backend",
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "image_sha256": pre.provenance.get("raw_bytes_sha256", ""),
            "source_path": str(img_path),
            "pipeline": pre.provenance,
            "disclaimer": DISCLAIMER_NON_DIAGNOSTIC,
            "explanatory_note": DISCLAIMER_NON_DIAGNOSTIC,
        }
        meta_path = dest / "metadata.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info("gradcam_written dir=%s layer=%s", dest, res.selected_layer)

    print(json.dumps({"run_id": run_id, "output_dir": str(out_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
