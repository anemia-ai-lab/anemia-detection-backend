#!/usr/bin/env python3
"""Descarga hand_landmarker.task para POST /predict multinail (MediaPipe Tasks)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_out = repo_root / "ml" / "artifacts" / "models" / "hand_landmarker.task"
    p = argparse.ArgumentParser(description="Descargar modelo Hand Landmarker (.task).")
    p.add_argument("--url", default=DEFAULT_URL, help="URL del modelo MediaPipe.")
    p.add_argument("--output", type=Path, default=default_out, help="Ruta de salida.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out: Path = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and out.stat().st_size > 0:
        print(f"Ya existe: {out}")
        return 0
    print(f"Descargando {args.url} -> {out}")
    try:
        urlretrieve(args.url, out)  # noqa: S310 — URL fija de Google MediaPipe
    except OSError as exc:
        print(f"Error al descargar: {exc}", file=sys.stderr)
        return 1
    print(f"Listo ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
