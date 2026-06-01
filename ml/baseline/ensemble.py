"""Promedio de probabilidades raw de varios checkpoints Keras."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from baseline.evaluation import collect_binary_predictions


def load_ensemble_models(paths: list[Path]) -> list[keras.Model]:
    models: list[keras.Model] = []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"No existe checkpoint ensemble: {p}")
        models.append(keras.models.load_model(p, compile=False))
    return models


def ensemble_raw_probabilities(
    models: list[keras.Model],
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Promedia ``raw_prob`` por modelo; ``y_true`` del primer modelo."""
    if not models:
        raise ValueError("ensemble requiere al menos un modelo")
    y_ref: np.ndarray | None = None
    prob_sum: np.ndarray | None = None
    for model in models:
        y_true, y_prob = collect_binary_predictions(model, dataset)
        if y_ref is None:
            y_ref = y_true
            prob_sum = y_prob.astype(np.float64, copy=True)
        else:
            if not np.array_equal(y_ref, y_true):
                raise ValueError("Los modelos del ensemble no alinean y_true en el dataset")
            prob_sum = prob_sum + y_prob.astype(np.float64, copy=False)
    assert y_ref is not None and prob_sum is not None
    return y_ref, prob_sum / float(len(models))
