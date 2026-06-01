#!/usr/bin/env python3
"""
Prepara el dataset Ghana (ROI uña, Mendeley) para entrenamiento/eval: resize 224×224 en train/test.

Raw típico: ``ml/data_raw/ghana/*.png`` (4260 con augmentación; ~507 sujetos en modo original-only).

Ejemplo::

    python ml/scripts/prepare_ghana_dataset.py \\
        --input-root ml/data_raw/ghana \\
        --output-dir ml/data/ghana
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_ML_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _ML_ROOT.parent

IMG_SIZE = 224

# Prefijos reales del zip Mendeley (typos del autor incluidos).
_PREFIX_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"^Anemic-FN-(\d+)", re.IGNORECASE), "positive", "ghana_fn"),
    (re.compile(r"^Non-Anrmic-FN-(\d+)", re.IGNORECASE), "negative", "ghana_fn"),
    (re.compile(r"^Anmeic-fn-(\d+)", re.IGNORECASE), "positive", "ghana_anmeic"),
    (re.compile(r"^Non-anemic-Fin-(\d+)", re.IGNORECASE), "negative", "ghana_fin"),
)

_CANONICAL_STEM = re.compile(
    r"^(Anemic-FN-|Non-Anrmic-FN-)(\d+)\.png$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GhanaEntry:
    path: Path
    label: str
    subject_key: str


def _repo_relative(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return "<fuera-del-repo>"


def parse_ghana_filename(name: str) -> tuple[str, str, str] | None:
    """
    Devuelve ``(label, subject_key, family)`` o ``None`` si no coincide con reglas conocidas.

    ``label`` es ``positive`` o ``negative``. ``subject_key`` incluye familia + id.
    """
    stem = Path(name).stem
    for rx, label, family in _PREFIX_RULES:
        m = rx.match(stem)
        if m:
            sid = m.group(1).lstrip("0") or "0"
            return label, f"{family}_{label}_{sid}", family
    return None


def is_canonical_original_stem(stem: str) -> bool:
    """``Anemic-FN-117.png`` sin espacios ni ``(N)``."""
    return _CANONICAL_STEM.match(f"{stem}.png") is not None


def pick_canonical_file(paths: list[Path]) -> Path:
    """Elige un PNG por sujeto: prioriza stem canónico FN-{id}.png; si no, el más corto."""
    canonical = [p for p in paths if is_canonical_original_stem(p.stem)]
    if canonical:
        return sorted(canonical, key=lambda p: (len(p.name), p.name))[0]
    return sorted(paths, key=lambda p: (len(p.name), p.name))[0]


def collect_entries(input_root: Path) -> list[GhanaEntry]:
    entries: list[GhanaEntry] = []
    for path in sorted(input_root.rglob("*.png")):
        parsed = parse_ghana_filename(path.name)
        if parsed is None:
            continue
        label, subject_key, _family = parsed
        entries.append(GhanaEntry(path=path, label=label, subject_key=subject_key))
    return entries


def group_by_subject(entries: list[GhanaEntry]) -> dict[str, list[GhanaEntry]]:
    by_subj: dict[str, list[GhanaEntry]] = defaultdict(list)
    for e in entries:
        by_subj[e.subject_key].append(e)
    return by_subj


def select_files_for_subjects(
    by_subj: dict[str, list[GhanaEntry]],
    *,
    original_only: bool,
) -> list[GhanaEntry]:
    out: list[GhanaEntry] = []
    for items in by_subj.values():
        if original_only:
            out.append(
                GhanaEntry(
                    path=pick_canonical_file([e.path for e in items]),
                    label=items[0].label,
                    subject_key=items[0].subject_key,
                )
            )
        else:
            out.extend(items)
    return out


def split_subjects(
    subject_keys: list[str],
    *,
    test_size: float,
    seed: int,
) -> dict[str, str]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size debe estar en (0, 1)")
    keys = list(subject_keys)
    rng = random.Random(seed)
    rng.shuffle(keys)
    if len(keys) <= 1:
        n_test = 0 if len(keys) == 0 else 1
    else:
        n_test = max(1, round(len(keys) * test_size))
    test_set = set(keys[:n_test])
    return {k: ("test" if k in test_set else "train") for k in keys}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepara Ghana ROI → PNG 224×224 train/test.")
    p.add_argument(
        "--input-root",
        type=Path,
        default=_ML_ROOT / "data_raw" / "ghana",
        help="Carpeta raw con PNG (default: ml/data_raw/ghana).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ML_ROOT / "data" / "ghana",
        help="Raíz de salida train/test/positive|negative.",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--include-augmented",
        action="store_true",
        help="Incluir variantes aumentadas (split por sujeto, varios PNG por sujeto).",
    )
    return p.parse_args()


def _resize_save_png_sips(src: Path, dst: Path) -> bool:
    """Fallback macOS: ``sips`` redimensiona sin importar TensorFlow."""
    import shutil
    import subprocess

    sips = shutil.which("sips")
    if not sips:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.png")
    shutil.copy2(src, tmp)
    r = subprocess.run(
        [sips, "-z", str(IMG_SIZE), str(IMG_SIZE), str(tmp), "--out", str(dst)],
        capture_output=True,
        text=True,
    )
    tmp.unlink(missing_ok=True)
    return r.returncode == 0 and dst.is_file()


def _resize_save_png_tf(src: Path, dst: Path) -> bool:
    import tensorflow as tf

    data = tf.io.read_file(str(src))
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    resized = tf.image.resize(
        tf.cast(image, tf.float32),
        [IMG_SIZE, IMG_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
    )
    out_u8 = tf.cast(tf.clip_by_value(tf.round(resized), 0, 255), tf.uint8)
    png = tf.io.encode_png(out_u8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(png.numpy())
    return True


def _resize_save_png(src: Path, dst: Path) -> bool:
    import sys

    if sys.platform == "darwin":
        if _resize_save_png_sips(src, dst):
            return True
    return _resize_save_png_tf(src, dst)


def main() -> None:
    args = _parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    original_only = not args.include_augmented

    if not input_root.is_dir():
        print(f"Error: no existe --input-root ({_repo_relative(input_root)})", file=sys.stderr)
        sys.exit(1)

    all_entries = collect_entries(input_root)
    if not all_entries:
        print(f"Error: no hay PNG reconocibles bajo {_repo_relative(input_root)}", file=sys.stderr)
        sys.exit(1)

    by_subj = group_by_subject(all_entries)
    selected = select_files_for_subjects(by_subj, original_only=original_only)
    split_map = split_subjects(
        sorted({e.subject_key for e in selected}),
        test_size=args.test_size,
        seed=args.seed,
    )

    family_counts: dict[str, int] = defaultdict(int)
    for e in all_entries:
        parsed = parse_ghana_filename(e.path.name)
        if parsed:
            family_counts[parsed[2]] += 1

    saved = 0
    crop_index: dict[tuple[str, str], int] = defaultdict(int)
    for entry in selected:
        split = split_map[entry.subject_key]
        idx = crop_index[(split, entry.subject_key)]
        crop_index[(split, entry.subject_key)] += 1
        safe_key = entry.subject_key.replace("/", "_")
        out_name = f"{safe_key}.png" if original_only else f"{safe_key}_{idx}.png"
        out_path = output_dir / split / entry.label / out_name
        try:
            _resize_save_png(entry.path, out_path)
            saved += 1
        except Exception as e:
            print(f"Aviso: omitido {entry.path.name} ({type(e).__name__})")

    train_subj = sum(1 for k, v in split_map.items() if v == "train")
    test_subj = sum(1 for k, v in split_map.items() if v == "test")
    n_pos = sum(1 for e in selected if e.label == "positive")
    n_neg = sum(1 for e in selected if e.label == "negative")

    print("--- Resumen Ghana ---")
    print(f"PNG reconocidos (raw): {len(all_entries)}")
    print(f"Familias raw: {dict(family_counts)}")
    print(f"Sujetos únicos: {len(by_subj)}")
    print(f"Modo: {'original-only' if original_only else 'include-augmented'}")
    print(f"Sujetos en split: train={train_subj}, test={test_subj}")
    print(f"Archivos escritos: {saved} (positive={n_pos}, negative={n_neg})")
    print(f"Salida: {_repo_relative(output_dir)}")


if __name__ == "__main__":
    main()
