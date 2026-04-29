"""Carga de datasets desde disco (o modo demo en memoria)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from baseline.config import (
    AUG_BRIGHTNESS_MAX_DELTA,
    AUG_CONTRAST_FACTOR,
    AUG_MAX_ROTATION_FACTOR,
    AUG_ZOOM_RANGE,
    BATCH_SIZE,
    IMG_SIZE,
    SEED,
)
from baseline.model import apply_mobilenet_preprocess
from tensorflow import keras
from tensorflow.keras import layers

ALLOWED_IMAGE_EXT: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"},
)


def _preprocess_batch(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    x = apply_mobilenet_preprocess(tf.cast(images, tf.float32))
    y = tf.reshape(tf.cast(labels, tf.float32), (-1, 1))
    return x, y


def _make_train_augmentation() -> keras.Sequential:
    """Augmentación moderada para imágenes médicas (solo train)."""
    return keras.Sequential(
        [
            layers.RandomRotation(AUG_MAX_ROTATION_FACTOR, fill_mode="reflect"),
            layers.RandomZoom(
                height_factor=AUG_ZOOM_RANGE,
                width_factor=AUG_ZOOM_RANGE,
                fill_mode="reflect",
            ),
            layers.RandomBrightness(AUG_BRIGHTNESS_MAX_DELTA),
            layers.RandomContrast(AUG_CONTRAST_FACTOR),
        ],
        name="train_augmentation_medical_light",
    )


def list_labeled_image_paths(train_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Lista rutas y etiquetas enteras en el mismo orden que ``image_dataset_from_directory``
    (subcarpetas de clase ordenadas alfabéticamente).
    """
    class_dirs = sorted([p for p in train_dir.iterdir() if p.is_dir()])
    if len(class_dirs) < 2:
        raise ValueError(f"Se esperan al menos 2 subcarpetas de clase en {train_dir}")
    class_names = [p.name for p in class_dirs]
    paths: list[str] = []
    labels: list[int] = []
    for label, d in enumerate(class_dirs):
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXT:
                paths.append(str(f.resolve()))
                labels.append(label)
    if not paths:
        raise ValueError(f"No hay imágenes válidas bajo {train_dir}")
    return np.array(paths, dtype=object), np.array(labels, dtype=np.int32), class_names


def stratified_train_val_paths(
    paths: np.ndarray,
    labels: np.ndarray,
    *,
    validation_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partición estratificada por clase (etiquetas 0/1)."""
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split debe estar en [0, 1).")
    rng = np.random.default_rng(seed)
    train_paths: list[str] = []
    train_labels: list[int] = []
    val_paths: list[str] = []
    val_labels: list[int] = []
    for c in (0, 1):
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        rng.shuffle(idx)
        n = int(idx.size)
        n_val = int(round(n * validation_split))
        if validation_split > 0.0 and n > 1:
            n_val = min(max(n_val, 1), n - 1)
        else:
            n_val = 0
        vidx, tidx = idx[:n_val], idx[n_val:]
        for i in vidx:
            val_paths.append(str(paths[i]))
            val_labels.append(int(labels[i]))
        for i in tidx:
            train_paths.append(str(paths[i]))
            train_labels.append(int(labels[i]))
    if validation_split > 0 and len(val_labels) == 0:
        raise ValueError(
            "El conjunto de validación quedó vacío (muy pocas imágenes por clase o "
            "validation_split demasiado bajo). Añade muestras o ajusta validation_split."
        )
    return (
        np.array(train_paths, dtype=object),
        np.array(train_labels, dtype=np.int32),
        np.array(val_paths, dtype=object),
        np.array(val_labels, dtype=np.int32),
    )


def count_by_class(labels: np.ndarray, *, n_classes: int = 2) -> dict[int, int]:
    out = {c: int(np.sum(labels == c)) for c in range(n_classes)}
    return out


def oversample_positive_train_balance(
    train_paths: np.ndarray,
    train_labels: np.ndarray,
    *,
    seed: int,
    positive_label: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """
    Duplica aleatoriamente (con reemplazo) muestras de la clase positiva hasta ``n_pos ≈ n_neg``
    en el subconjunto de entrenamiento. No altera validación ni test.
    """
    y = np.asarray(train_labels, dtype=np.int32)
    p = np.asarray(train_paths, dtype=object)
    n_pos = int(np.sum(y == positive_label))
    n_neg = int(np.sum(y != positive_label))
    before = {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))}
    if n_pos == 0:
        return p, y, {
            "applied": False,
            "reason": "no_positive_samples",
            "before_train_by_class": dict(before),
            "after_train_by_class": dict(before),
            "duplicates_added": 0,
            "positive_label": positive_label,
        }
    if n_pos >= n_neg:
        return p, y, {
            "applied": False,
            "reason": "already_balanced_or_majority_positive",
            "before_train_by_class": dict(before),
            "after_train_by_class": dict(before),
            "duplicates_added": 0,
            "positive_label": positive_label,
        }
    n_add = n_neg - n_pos
    rng = np.random.default_rng(seed)
    idx_pos = np.flatnonzero(y == positive_label)
    pick = rng.choice(idx_pos, size=n_add, replace=True)
    extra_p = p[pick].copy()
    extra_y = y[pick].copy()
    new_p = np.concatenate([p, extra_p])
    new_y = np.concatenate([y, extra_y])
    perm = rng.permutation(new_y.size)
    new_p, new_y = new_p[perm], new_y[perm]
    after = {"0": int(np.sum(new_y == 0)), "1": int(np.sum(new_y == 1))}
    return new_p, new_y, {
        "applied": True,
        "strategy": (
            "Oversampling aleatorio con reemplazo de la clase positiva (label=1) hasta "
            "aproximadamente 1:1 con negativos; solo en el subconjunto train del split interno."
        ),
        "before_train_by_class": dict(before),
        "after_train_by_class": dict(after),
        "duplicates_added": int(n_add),
        "positive_label": positive_label,
    }


def compute_binary_class_weights(counts: dict[int, int]) -> dict[int, float]:
    """Pesos balanceados tipo ``balanced``: n / (k * n_c), k=2 clases."""
    n_classes = 2
    total = sum(max(0, counts.get(c, 0)) for c in range(n_classes))
    if total == 0:
        return {0: 1.0, 1: 1.0}
    weights: dict[int, float] = {}
    for c in range(n_classes):
        n_c = max(counts.get(c, 0), 1)
        weights[c] = float(total / (n_classes * n_c))
    return weights


def count_images_in_class_folders(data_dir: Path) -> dict[str, int]:
    """Cuenta imágenes por nombre de subcarpeta (clase)."""
    out: dict[str, int] = {}
    for sub in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        n = sum(1 for f in sub.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXT)
        out[sub.name] = n
    return out


def count_total_crops(*dirs: Path) -> int:
    t = 0
    for d in dirs:
        if not d.is_dir():
            continue
        for sub in d.iterdir():
            if sub.is_dir():
                t += sum(
                    1 for f in sub.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXT
                )
    return t


def count_unique_patients_from_crops(*dirs: Path) -> int:
    """
    Heurística para crops ``{patient_id}_{i}.png``: agrupa por prefijo antes del último ``_``
    si el sufijo es numérico.
    """
    pids: set[str] = set()
    for d in dirs:
        if not d.is_dir():
            continue
        for img in d.rglob("*"):
            if not img.is_file() or img.suffix.lower() not in ALLOWED_IMAGE_EXT:
                continue
            stem = img.stem
            if "_" in stem:
                head, tail = stem.rsplit("_", 1)
                if tail.isdigit():
                    pids.add(head)
                    continue
            pids.add(stem)
    return len(pids)


def read_unique_patient_count_from_metadata(metadata_path: Path | None) -> int | None:
    if metadata_path is None or not metadata_path.is_file():
        return None
    ids: set[str] = set()
    with metadata_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "PATIENT_ID" not in reader.fieldnames:
            return None
        for row in reader:
            pid = str(row.get("PATIENT_ID", "")).strip()
            if pid:
                ids.add(pid)
    return len(ids) if ids else None


def load_train_val_datasets(
    train_dir: Path,
    *,
    validation_split: float = 0.2,
    batch_size: int = BATCH_SIZE,
    img_size: tuple[int, int] = IMG_SIZE,
    seed: int = SEED,
    augment_train: bool = True,
    oversample_positive: bool = False,
) -> tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    dict[int, int],
    dict[int, int],
    dict[int, float],
    list[str],
    dict[str, object],
]:
    """
    Train/val desde ``train_dir`` con split estratificado reproducible.

    - Los pesos de clase se calculan con los conteos del **train** tras el split y **antes**
      de cualquier oversampling (distribución original del split).
    - Si ``oversample_positive`` es True, se duplican muestras positivas solo en train hasta ~1:1.
    - ``val_counts`` refleja el subconjunto de validación del mismo split.
    - Augmentación solo en train si ``augment_train`` es True.
    """
    paths, labels, class_names = list_labeled_image_paths(train_dir)
    train_paths, train_labels, val_paths, val_labels = stratified_train_val_paths(
        paths,
        labels,
        validation_split=validation_split,
        seed=seed,
    )
    train_counts = count_by_class(train_labels)
    val_counts = count_by_class(val_labels)
    class_weight = compute_binary_class_weights(train_counts)
    oversampling_info: dict[str, object]
    if oversample_positive:
        train_paths, train_labels, oversampling_info = oversample_positive_train_balance(
            train_paths,
            train_labels,
            seed=seed,
        )
    else:
        oversampling_info = {
            "applied": False,
            "reason": "oversample_positive_disabled",
            "before_train_by_class": {str(k): int(v) for k, v in train_counts.items()},
            "after_train_by_class": {str(k): int(v) for k, v in train_counts.items()},
            "duplicates_added": 0,
        }
    aug = _make_train_augmentation() if augment_train else None

    def _decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        data = tf.io.read_file(path)
        img = tf.io.decode_image(data, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, list(img_size))
        img = tf.cast(img, tf.float32)
        return img, label

    def _train_batch_augment(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        assert aug is not None
        images = aug(images, training=True)
        images = tf.clip_by_value(images, 0.0, 255.0)
        return _preprocess_batch(images, labels)

    def _batch_preprocess_only(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return _preprocess_batch(images, labels)

    shuffle_buf = min(int(train_paths.size), 2048)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .shuffle(shuffle_buf, seed=seed, reshuffle_each_iteration=True)
        .map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
    )
    if augment_train:
        train_ds = train_ds.map(
            _train_batch_augment,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        train_ds = train_ds.map(
            _batch_preprocess_only,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        .map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .map(_batch_preprocess_only, num_parallel_calls=tf.data.AUTOTUNE)
    )

    return (
        train_ds.prefetch(tf.data.AUTOTUNE),
        val_ds.prefetch(tf.data.AUTOTUNE),
        train_counts,
        val_counts,
        class_weight,
        class_names,
        oversampling_info,
    )


def load_validation_dataset(
    train_dir: Path,
    *,
    validation_split: float = 0.2,
    batch_size: int = BATCH_SIZE,
    img_size: tuple[int, int] = IMG_SIZE,
    seed: int = SEED,
) -> tuple[tf.data.Dataset, dict[int, int], list[str]]:
    """
    Subconjunto de **validación** reproducible (mismo split estratificado que
    ``load_train_val_datasets``), sin augmentación ni oversampling.

    Útil para calibración post-hoc (p. ej. *temperature scaling*) sin reconstruir train.
    """
    paths, labels, class_names = list_labeled_image_paths(train_dir)
    _, _, val_paths, val_labels = stratified_train_val_paths(
        paths,
        labels,
        validation_split=validation_split,
        seed=seed,
    )
    val_counts = count_by_class(val_labels)

    def _decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        data = tf.io.read_file(path)
        img = tf.io.decode_image(data, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, list(img_size))
        img = tf.cast(img, tf.float32)
        return img, label

    def _batch_preprocess_only(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return _preprocess_batch(images, labels)

    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        .map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .map(_batch_preprocess_only, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return val_ds, val_counts, class_names


def load_image_datasets(
    train_dir: Path,
    *,
    validation_split: float = 0.2,
    batch_size: int = BATCH_SIZE,
    img_size: tuple[int, int] = IMG_SIZE,
    seed: int = SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Compatibilidad: train/val sin augmentación (mismo split estratificado)."""
    train_ds, val_ds, _, _, _, _, _ = load_train_val_datasets(
        train_dir,
        validation_split=validation_split,
        batch_size=batch_size,
        img_size=img_size,
        seed=seed,
        augment_train=False,
        oversample_positive=False,
    )
    return train_ds, val_ds


def load_test_dataset(
    test_dir: Path,
    *,
    batch_size: int = BATCH_SIZE,
    img_size: tuple[int, int] = IMG_SIZE,
) -> tf.data.Dataset:
    ds = keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )
    return ds.map(_preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE,
    )


def make_demo_datasets(
    *,
    n_batches: int = 8,
    batch_size: int = BATCH_SIZE,
    img_size: tuple[int, int] = IMG_SIZE,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Datos sintéticos para comprobar que el pipeline y el modelo encajan."""

    def _batch(_: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        images = tf.random.uniform(
            (batch_size, *img_size, 3),
            0,
            255,
            dtype=tf.float32,
        )
        labels = tf.random.uniform((batch_size,), 0, 2, dtype=tf.int32)
        return _preprocess_batch(images, labels)

    ds = (
        tf.data.Dataset.range(n_batches)
        .map(_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds, ds


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
