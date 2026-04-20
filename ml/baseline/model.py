"""Arquitectura MobileNetV2 + cabezal denso (clasificación binaria)."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

from baseline.config import IMG_SIZE


def build_model(
    *,
    img_size: tuple[int, int] = IMG_SIZE,
    dropout: float = 0.2,
    backbone_trainable: bool = False,
) -> keras.Model:
    """MobileNetV2 (ImageNet) sin top + GAP + Dropout + Dense(1, sigmoid)."""
    inputs = keras.Input(shape=(*img_size, 3), name="image")
    base = MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet",
        name="mobilenet_backbone",
    )
    base.trainable = backbone_trainable
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_baseline_binary")


def set_backbone_trainable(model: keras.Model, freeze_up_to_layer: int) -> None:
    """Congela capas del backbone hasta el índice dado; el resto queda entrenable."""
    base = model.get_layer("mobilenet_backbone")
    base.trainable = True
    if freeze_up_to_layer == 0:
        return
    if freeze_up_to_layer < 0:
        cutoff = len(base.layers) + freeze_up_to_layer
    else:
        cutoff = freeze_up_to_layer
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= max(0, cutoff)


def compile_for_binary(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def apply_mobilenet_preprocess(images: tf.Tensor) -> tf.Tensor:
    """Escala a rango esperado por MobileNetV2 ([-1, 1])."""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    return preprocess_input(images)
