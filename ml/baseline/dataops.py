"""Carga de datasets desde disco (o modo demo en memoria)."""

from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from baseline.config import BATCH_SIZE, IMG_SIZE, SEED
from baseline.model import apply_mobilenet_preprocess


def _preprocess_batch(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    x = apply_mobilenet_preprocess(tf.cast(images, tf.float32))
    y = tf.reshape(tf.cast(labels, tf.float32), (-1, 1))
    return x, y


def load_image_datasets(
    train_dir: Path,
    *,
    validation_split: float = 0.2,
    batch_size: int = BATCH_SIZE,
    img_size: tuple[int, int] = IMG_SIZE,
    seed: int = SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Dos subcarpetas de clases bajo ``train_dir`` (p. ej. negative/ positive)."""
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
    )
    train_ds = train_ds.map(_preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(_preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return (
        train_ds.prefetch(tf.data.AUTOTUNE),
        val_ds.prefetch(tf.data.AUTOTUNE),
    )


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
    # train / val simulados (mismo generador; solo smoke test)
    return ds, ds


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
