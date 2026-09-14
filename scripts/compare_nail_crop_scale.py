#!/usr/bin/env python3
"""Compara ``crop_scale`` 1.0 vs 0.75 en fotos reales (ops; no es test de CI).

El test unitario geométrico solo comprueba que el bbox es más chico. Este script
pasa fotos por detección (+ inferencia si hay modelo) y reporta ``p_cal`` por uña.

Si 0.75 no reduce positivos de forma clara, dejar ``PREDICT_NAIL_CROP_SCALE=1.0``.
Esto no cierra el domain gap (Etapa 3: fine-tuning con fotos tipo-producción).

Uso::

    PYTHONPATH=. python scripts/compare_nail_crop_scale.py foto1.jpg foto2.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comparar crop_scale 1.0 vs 0.75 en fotos.")
    p.add_argument("images", nargs="+", type=Path, help="JPEG/PNG de mano (captura tipo-app).")
    p.add_argument("--scales", default="1.0,0.75", help="Lista de escalas, coma-separada.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    scales = [float(x.strip()) for x in args.scales.split(",") if x.strip()]
    from backend.core.config import settings
    from backend.inference.nail_detection import MediaPipeNailDetector
    from backend.inference.prediction_image_input import prepare_prediction_image
    from backend.inference.probability_calibration import apply_probability_calibration

    detector_by_scale = {
        s: MediaPipeNailDetector(crop_scale=s) for s in scales
    }
    predictor = None
    try:
        from backend.inference.runtime import get_builtin_image_predictor

        predictor = get_builtin_image_predictor()
    except Exception as exc:  # noqa: BLE001 — ops script
        print(f"Aviso: sin predictor ({type(exc).__name__}: {exc}). Solo bbox.", file=sys.stderr)

    any_ok = False
    for image_path in args.images:
        path = image_path.expanduser().resolve()
        if not path.is_file():
            print(f"No existe: {path}", file=sys.stderr)
            continue
        any_ok = True
        print(f"\n=== {path.name} ===")
        suffix = path.suffix.lower()
        content_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")
        _ct, _proc, rgb = prepare_prediction_image(content_type, path.read_bytes())
        for scale in scales:
            crops = detector_by_scale[scale].detect(rgb)
            print(f"  crop_scale={scale}: {len(crops)} uñas")
            for crop in crops:
                w, h = int(crop.bbox[2]), int(crop.bbox[3])
                extra = ""
                if predictor is not None:
                    raw = float(predictor.predict_from_rgb(crop.rgb))
                    cal = apply_probability_calibration(
                        raw,
                        method=str(settings.inference_calibration_method),
                        temperature=float(settings.inference_calibration_temperature),
                        platt_a=float(settings.inference_calibration_platt_a),
                        platt_b=float(settings.inference_calibration_platt_b),
                    )
                    extra = f" raw={raw:.4f} cal={cal:.4f}"
                print(f"    {crop.finger}: bbox={w}x{h}{extra}")
    if not any_ok:
        return 1
    print(
        "\nInterpretación: si 0.75 no baja cal/positivos vs 1.0, no cambiar el default. "
        "Median aggregation mitiga max; el domain gap queda para Etapa 3."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
