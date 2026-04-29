"""
Grad-CAM (G10) para el modelo MobileNetV2 baseline.

Usa el mismo preprocesado G9 y la misma calibración por temperatura que el backend.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ML_ROOT = Path(__file__).resolve().parents[1]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from backend.core.risk_mapping import anemia_risk_label, risk_from_probability
from backend.inference.probability_calibration import (
    apply_temperature_calibration,
    binary_prediction_from_threshold,
)
from ml.preprocessing.pipeline import PreprocessingConfig, preprocess_rgb_array

logger = logging.getLogger(__name__)

# Nombre del backbone en ``ml/baseline/model.py`` (mantener alineado).
_MOBILENET_BACKBONE_NAME = "mobilenet_backbone"


class GradCAMError(RuntimeError):
    """Fallo al construir o ejecutar Grad-CAM (gradientes, forma de capa, etc.)."""


def _jet_colormap_numpy(values_01: np.ndarray) -> np.ndarray:
    """Jet aproximado en [0,1]^3 sin dependencias externas (``values_01`` en [0,1], shape H×W)."""
    v = np.clip(values_01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _try_matplotlib_jet(values_01: np.ndarray) -> np.ndarray | None:
    try:
        from matplotlib import colormaps

        jet = colormaps["jet"]
        return jet(values_01)[..., :3].astype(np.float32)
    except Exception:
        return None


def _heatmap_to_rgb(heatmap_01: np.ndarray) -> np.ndarray:
    rgb = _try_matplotlib_jet(heatmap_01)
    if rgb is None:
        rgb = _jet_colormap_numpy(heatmap_01)
    return rgb


def _find_last_conv_in_backbone(backbone: Any) -> Any:
    from tensorflow import keras

    last: Any = None
    for sub in backbone.layers:
        if isinstance(sub, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
            last = sub
    if last is None:
        raise GradCAMError("no se encontró capa Conv2D/DepthwiseConv2D en el backbone")
    return last


def _get_layer_resolving_backbone_prefix(model: Any, name: str) -> Any:
    """Busca en el modelo, con prefijo ``mobilenet_backbone/``, o bajo el backbone (Keras 3)."""
    last_err: ValueError | None = None
    for candidate in (name, f"{_MOBILENET_BACKBONE_NAME}/{name}"):
        try:
            return model.get_layer(candidate)
        except ValueError as e:
            last_err = e
            continue
    try:
        bb = model.get_layer(_MOBILENET_BACKBONE_NAME)
        return bb.get_layer(name)
    except ValueError as e:
        last_err = e
    assert last_err is not None
    raise GradCAMError(f"capa no encontrada: {name}") from last_err


def _same_named_conv_in_backbone(backbone: Any, ref: Any) -> Any:
    """Capa conv homónima en ``backbone.layers`` (mismo grafo que ``model.input``)."""
    from tensorflow import keras

    target = ref.name
    for sub in backbone.layers:
        if (
            sub.name == target
            and isinstance(sub, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D))
        ):
            return sub
    return ref


def _select_gradcam_conv_layer(model: Any, *, layer_name: str | None) -> Any:
    from tensorflow import keras

    try:
        backbone = model.get_layer(_MOBILENET_BACKBONE_NAME)
    except ValueError as exc:
        if layer_name is None:
            raise GradCAMError(
                f"no se encontró backbone '{_MOBILENET_BACKBONE_NAME}'; use --layer explícito",
            ) from exc
        backbone = None

    if layer_name is None:
        assert backbone is not None
        return _find_last_conv_in_backbone(backbone)

    lyr = _get_layer_resolving_backbone_prefix(model, layer_name)
    if isinstance(lyr, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
        if backbone is not None:
            return _same_named_conv_in_backbone(backbone, lyr)
        return lyr
    if hasattr(lyr, "layers"):
        return _find_last_conv_in_backbone(lyr)
    msg = f"capa '{layer_name}' no es convolucional ni un contenedor con sub-capas conv"
    raise GradCAMError(msg)


@dataclass
class GradCAMResult:
    preprocessed: np.ndarray
    heatmap: np.ndarray
    overlay: np.ndarray
    saliency: np.ndarray
    raw_probability: float
    calibrated_probability: float
    selected_layer: str
    explanation_status: str


class GradCAM:
    """Grad-CAM sobre la salida escalar (probabilidad sigmoide) del modelo binario."""

    def __init__(
        self,
        model: Any,
        *,
        layer_name: str | None = None,
        temperature: float | None = None,
        operational_threshold: float | None = None,
    ) -> None:
        from tensorflow import keras

        from backend.core.config import settings

        self._model = model
        try:
            backbone = model.get_layer(_MOBILENET_BACKBONE_NAME)
        except ValueError:
            backbone = None

        conv_layer = _select_gradcam_conv_layer(model, layer_name=layer_name)
        self._selected = conv_layer.name
        if not isinstance(conv_layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
            msg = f"capa seleccionada no es convolucional: {self._selected}"
            raise GradCAMError(msg)
        try:
            if backbone is not None and any(
                sub is conv_layer for sub in backbone.layers
            ):
                # Keras 3: ``model.input`` y ``backbone.input`` son tensores distintos.
                bb_conv = keras.Model(
                    inputs=backbone.input,
                    outputs=conv_layer.output,
                    name="gradcam_backbone_conv",
                )
                self._grad_model = keras.Model(
                    inputs=model.input,
                    outputs=[bb_conv(model.input), model.output],
                    name="gradcam_aux",
                )
            else:
                self._grad_model = keras.Model(
                    inputs=model.input,
                    outputs=[conv_layer.output, model.output],
                    name="gradcam_aux",
                )
        except ValueError as exc:
            raise GradCAMError("no se pudo construir el modelo auxiliar Grad-CAM") from exc

        self._temperature = float(
            temperature if temperature is not None else settings.inference_calibration_temperature,
        )
        self._threshold = float(
            operational_threshold
            if operational_threshold is not None
            else settings.inference_calibration_operational_threshold,
        )

    @property
    def selected_layer(self) -> str:
        return self._selected

    def explain(
        self,
        rgb_uint8: np.ndarray,
        *,
        cfg: PreprocessingConfig | None = None,
    ) -> GradCAMResult:
        import tensorflow as tf

        cfg = cfg or PreprocessingConfig()
        pre = preprocess_rgb_array(rgb_uint8, cfg=cfg)
        batch = tf.constant(pre.model_input_tensor, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(batch)
            conv_out, preds = self._grad_model(batch, training=False)
            pred_vec = preds[:, 0]
            loss = pred_vec[0]

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            # Keras 3 + grafo auxiliar ``bb_conv(model.input)``: la predicción no retropropaga
            # al tensor ``conv_out`` del submodelo. Se usa un mapa neutro (canal medio) para no
            # abortar; no sustituye Grad-CAM fiel hasta unificar el grafo en una sola pasada.
            explanation_status = "degraded_no_gradients"
            g_np = np.ones_like(conv_out.numpy(), dtype=np.float32) / float(
                max(1, conv_out.shape[-1]),
            )
        else:
            explanation_status = "gradcam"
            g_np = grads.numpy()
        conv_np = conv_out.numpy()
        weights = np.mean(g_np, axis=(1, 2))[0]  # (C,)
        cam = np.zeros(conv_np.shape[1:3], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += float(w) * conv_np[0, :, :, i]
        cam = np.maximum(cam, 0.0)
        mx = float(np.max(cam)) if cam.size else 0.0
        if mx < 1e-12:
            heat_small = np.zeros_like(cam, dtype=np.float32)
        else:
            heat_small = cam / mx

        h224 = tf.image.resize(np.expand_dims(heat_small, -1), [224, 224], method="bilinear")
        heatmap = tf.squeeze(h224, axis=-1).numpy().astype(np.float32)

        rgb224 = (
            tf.cast(
                tf.image.resize(pre.decoded_rgb_uint8, [224, 224]),
                tf.float32,
            )
            .numpy()
        )
        heat_rgb = _heatmap_to_rgb(heatmap)
        alpha = 0.45
        blend = alpha * heat_rgb + (1.0 - alpha) * (rgb224 / 255.0)
        overlay = np.clip(np.round(blend * 255.0), 0, 255).astype(np.uint8)

        saliency = np.clip(np.round(heatmap * 255.0), 0, 255).astype(np.uint8)

        raw_p = float(loss.numpy())
        raw_p = max(0.0, min(1.0, raw_p))
        cal = apply_temperature_calibration(raw_p, self._temperature)

        return GradCAMResult(
            preprocessed=pre.model_input_tensor.copy(),
            heatmap=heatmap,
            overlay=overlay,
            saliency=saliency,
            raw_probability=raw_p,
            calibrated_probability=cal,
            selected_layer=self._selected,
            explanation_status=explanation_status,
        )

    def explain_with_decision(
        self,
        rgb_uint8: np.ndarray,
        *,
        cfg: PreprocessingConfig | None = None,
    ) -> tuple[GradCAMResult, int, str, str]:
        """``(result, prediction, risk, risk_label)`` con umbral operacional del backend."""
        r = self.explain(rgb_uint8, cfg=cfg)
        pred = binary_prediction_from_threshold(r.calibrated_probability, self._threshold)
        risk = risk_from_probability(r.calibrated_probability, self._threshold)
        label = anemia_risk_label(risk)
        return r, pred, risk, label
