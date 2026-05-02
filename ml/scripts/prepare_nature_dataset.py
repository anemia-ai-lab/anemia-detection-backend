#!/usr/bin/env python3
"""
Prepara el dataset Nature (uñas) para entrenamiento CNN: crops 224×224 en train/test por clase.

Ejemplo (este repo usa ``data_raw`` con guión bajo)::

    python ml/scripts/prepare_nature_dataset.py \\
        --images-dir ml/data_raw/nature/images \\
        --metadata-path ml/data_raw/nature/metadata.csv

Por defecto se usan ``ml/data_raw/...``; si no existen, pasa explícitamente
``--images-dir`` / ``--metadata-path``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_ML_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _ML_ROOT.parent


def _repo_relative(p: Path) -> str:
    """Solo para mensajes: ruta relativa al repo; evita exponer rutas absolutas de usuario."""
    try:
        return str(p.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return "<fuera-del-repo>"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genera crops 224×224 (PNG) desde Nature metadata + imágenes.",
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=_ML_ROOT / "data_raw" / "nature" / "images",
        help="Carpeta con imágenes {PATIENT_ID}.jpg (default: ml/data_raw/nature/images).",
    )
    p.add_argument(
        "--metadata-path",
        type=Path,
        default=_ML_ROOT / "data_raw" / "nature" / "metadata.csv",
        help="CSV con PATIENT_ID, HB_LEVEL_GperL, NAIL_BOUNDING_BOXES.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ML_ROOT / "data",
        help="Raíz bajo la que se crean train/test/positive|negative.",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fracción de pacientes en test (0–1).",
    )
    p.add_argument(
        "--sample-patients",
        type=int,
        default=None,
        metavar="N",
        help="Si se indica, solo se procesan N pacientes elegidos al azar.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reparto train/test y muestreo.",
    )
    return p.parse_args()


def _label_from_hb(hb: float) -> str:
    return "positive" if hb < 110.0 else "negative"


def _parse_boxes_list(raw: str) -> list[list[int]]:
    """``ast.literal_eval`` de la lista; cada caja es ``[y1, x1, y2, x2]``; omite cajas inválidas."""
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        data = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as e:
        print(f"Aviso: NAIL_BOUNDING_BOXES no parseable ({type(e).__name__}).")
        return []
    if not isinstance(data, list):
        print("Aviso: cajas no son una lista tras parsear.")
        return []
    out: list[list[int]] = []
    for i, item in enumerate(data):
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            print(f"Aviso: caja índice {i} ignorada (no tiene 4 enteros).")
            continue
        try:
            # metadata: [y1, x1, y2, x2]
            out.append([int(item[0]), int(item[1]), int(item[2]), int(item[3])])
        except (TypeError, ValueError):
            print(f"Aviso: caja índice {i} ignorada (enteros inválidos).")
    return out


def _clamp_box(x1: int, y1: int, x2: int, y2: int, w_img: int, h_img: int) -> tuple[int, int, int, int] | None:
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(0, min(x2, w_img))
    y2 = max(0, min(y2, h_img))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _find_image(images_dir: Path, patient_id: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"):
        p = images_dir / f"{patient_id}{ext}"
        if p.is_file():
            return p
    return None


def _load_image(path: Path) -> Any:
    import tensorflow as tf

    data = tf.io.read_file(str(path))
    return tf.io.decode_image(data, channels=3, expand_animations=False)


def _crop_resize_save(
    image_u8: Any,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    out_path: Path,
) -> bool:
    import tensorflow as tf

    h_i = int(tf.shape(image_u8)[0].numpy())
    w_i = int(tf.shape(image_u8)[1].numpy())
    box = _clamp_box(x1, y1, x2, y2, w_i, h_i)
    if box is None:
        return False
    x1, y1, x2, y2 = box
    ch = y2 - y1
    cw = x2 - x1
    if cw < 1 or ch < 1:
        return False
    cropped = tf.image.crop_to_bounding_box(image_u8, y1, x1, ch, cw)
    resized = tf.image.resize(
        tf.cast(cropped, tf.float32),
        [224, 224],
        method=tf.image.ResizeMethod.BILINEAR,
    )
    out_u8 = tf.cast(tf.clip_by_value(tf.round(resized), 0, 255), tf.uint8)
    png = tf.io.encode_png(out_u8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png.numpy())
    return True


def main() -> None:
    args = _parse_args()
    images_dir = args.images_dir.expanduser().resolve()
    metadata_path = args.metadata_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    test_size = args.test_size
    if not 0.0 < test_size < 1.0:
        print("--test-size debe estar entre 0 y 1 (excluidos).", file=sys.stderr)
        sys.exit(1)

    if not metadata_path.is_file():
        print(f"Error: no existe metadata ({_repo_relative(metadata_path)})", file=sys.stderr)
        sys.exit(1)
    if not images_dir.is_dir():
        print(f"Error: no existe carpeta de imágenes ({_repo_relative(images_dir)})", file=sys.stderr)
        sys.exit(1)

    import tensorflow as tf

    rows: list[dict[str, str]] = []
    with metadata_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"PATIENT_ID", "HB_LEVEL_GperL", "NAIL_BOUNDING_BOXES"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            print(
                f"CSV debe incluir columnas {sorted(required)}; hay: {reader.fieldnames}",
                file=sys.stderr,
            )
            sys.exit(1)
        for row in reader:
            rows.append(row)

    patient_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        pid = str(row["PATIENT_ID"]).strip()
        if pid and pid not in seen:
            seen.add(pid)
            patient_ids.append(pid)

    if not patient_ids:
        print("Error: no hay PATIENT_ID en el CSV.", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    if args.sample_patients is not None:
        n = max(0, min(args.sample_patients, len(patient_ids)))
        patient_ids = rng.sample(patient_ids, n) if n < len(patient_ids) else list(patient_ids)

    rng.shuffle(patient_ids)

    n_test = 0
    if len(patient_ids) > 1:
        n_test = max(1, int(round(len(patient_ids) * test_size)))
    test_set = set(patient_ids[:n_test])
    train_set = set(patient_ids[n_test:])

    split_by_patient: dict[str, str] = {}
    for pid in train_set:
        split_by_patient[pid] = "train"
    for pid in test_set:
        split_by_patient[pid] = "test"

    crop_index: dict[str, int] = defaultdict(int)
    crops_saved = 0
    skipped = 0
    patients_with_crop: set[str] = set()

    for row in rows:
        pid = str(row["PATIENT_ID"]).strip()
        if pid not in split_by_patient:
            continue

        hb_raw = row.get("HB_LEVEL_GperL", "").strip()
        try:
            hb = float(hb_raw)
        except ValueError:
            print("Aviso: HB_LEVEL_GperL no numérico; fila omitida.")
            skipped += 1
            continue

        label = _label_from_hb(hb)
        boxes = _parse_boxes_list(row.get("NAIL_BOUNDING_BOXES", ""))
        if not boxes:
            print("Aviso: fila sin cajas válidas; omitida.")
            skipped += 1
            continue

        img_path = _find_image(images_dir, pid)
        if img_path is None:
            print("Aviso: imagen de entrada no encontrada; fila omitida.")
            skipped += 1
            continue

        try:
            image_tf = _load_image(img_path)
            image_tf.set_shape([None, None, 3])
            image_u8 = tf.cast(image_tf, tf.uint8)
        except Exception as e:
            print(f"Aviso: error al cargar imagen ({type(e).__name__})")
            skipped += 1
            continue

        split = split_by_patient[pid]
        out_sub = output_dir / split / label

        for box in boxes:
            # metadata: [y1, x1, y2, x2] → entrada de recorte: (x1, y1, x2, y2)
            y1_box, x1_box, y2_box, x2_box = box
            x1, y1, x2, y2 = x1_box, y1_box, x2_box, y2_box
            idx = crop_index[pid]
            out_path = out_sub / f"{pid}_{idx}.png"
            crop_index[pid] += 1
            try:
                ok = _crop_resize_save(image_u8, x1, y1, x2, y2, out_path)
            except Exception as e:
                print(f"Aviso: error al guardar crop ({type(e).__name__})")
                skipped += 1
                continue
            if not ok:
                print("Aviso: crop inválido o vacío; omitido.")
                skipped += 1
                continue
            crops_saved += 1
            patients_with_crop.add(pid)

    print("--- Resumen ---")
    print(f"Pacientes en reparto (train/test): {len(patient_ids)}")
    print(f"Pacientes con al menos un PNG guardado: {len(patients_with_crop)}")
    print(f"Crops guardados: {crops_saved}")
    print(f"Ítems omitidos (avisos / filas): {skipped}")
    print(f"Salida (--output-dir): {_repo_relative(output_dir)}")


if __name__ == "__main__":
    main()
