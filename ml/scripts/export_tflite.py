#!/usr/bin/env python3
"""
Exporta el modelo final MobileNetV2 (.keras) a TensorFlow Lite (.tflite).

La inferencia en backend FastAPI sigue usando el .keras; este artefacto sirve para
despliegues móviles **offline** con el mismo grafo de salida (probabilidad sigmoide).
La calibración por temperatura y el umbral operacional deben aplicarse **en el cliente**
usando los valores del JSON de metadatos generado junto al .tflite.

Uso::

    cd ml && pip install -r requirements.txt
    python scripts/export_tflite.py

    python scripts/export_tflite.py \\
        --keras-path artifacts/models/baseline_mobilenetv2.keras \\
        --output-tflite artifacts/models/baseline_mobilenetv2_v1.tflite \\
        --output-metadata artifacts/models/baseline_mobilenetv2_v1.metadata.json \\
        --overwrite

Flags::
    --verify              Validación post-conversión (predeterminado).
    --no-verify           Omitir validación (depuración).
    --overwrite           Sobrescribe ``.tflite`` / JSON si ya existen (sin esto, falla antes de convertir).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from baseline.config import DEFAULT_MODEL_NAME, IMG_SIZE, MODEL_DIR, SEED  # noqa: E402
from tensorflow import keras  # noqa: E402

DEFAULT_TFLITE_NAME = "baseline_mobilenetv2_v1.tflite"
DEFAULT_METADATA_NAME = "baseline_mobilenetv2_v1.metadata.json"

MODEL_VERSION = "v1.0"
TEMPERATURE = 0.7510018331928743
OPERATIONAL_THRESHOLD = 0.1680544387290045
PREPROCESSING = "mobilenet_v2.preprocess_input"

EXPECTED_BATCH = 1
EXPECTED_HW = (IMG_SIZE[0], IMG_SIZE[1])


class ConversionError(RuntimeError):
    """Fallo al convertir el modelo Keras a TensorFlow Lite."""


class VerificationError(RuntimeError):
    """Fallo en la validación del modelo TensorFlow Lite exportado."""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exportar baseline .keras a TensorFlow Lite.")
    p.add_argument(
        "--keras-path",
        type=Path,
        default=MODEL_DIR / DEFAULT_MODEL_NAME,
        help="Archivo .keras entrenado.",
    )
    p.add_argument(
        "--output-tflite",
        type=Path,
        default=MODEL_DIR / DEFAULT_TFLITE_NAME,
        help="Ruta del .tflite de salida (debe estar bajo ml/artifacts/models/).",
    )
    p.add_argument(
        "--output-metadata",
        type=Path,
        default=MODEL_DIR / DEFAULT_METADATA_NAME,
        help="JSON de metadatos (debe estar bajo ml/artifacts/models/).",
    )
    p.set_defaults(verify=True)
    verify_group = p.add_mutually_exclusive_group()
    verify_group.add_argument(
        "--verify",
        dest="verify",
        action="store_true",
        help="Validar el .tflite tras la conversión (predeterminado).",
    )
    verify_group.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="No ejecutar validación post-export.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribe salidas si ya existen (sin este flag, falla si hay conflicto).",
    )
    return p.parse_args()


def _resolved_models_root() -> Path:
    return MODEL_DIR.resolve()


def _assert_path_under_models_dir(path: Path, *, label: str) -> Path:
    root = _resolved_models_root()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as e:
        raise ValueError(
            f"{label} debe escribirse bajo {root}; recibido: {resolved}"
        ) from e
    return resolved


def _check_outputs_exist(paths: list[Path], *, overwrite: bool) -> None:
    conflicts = [p for p in paths if p.exists()]
    if conflicts and not overwrite:
        listed = ", ".join(str(p) for p in conflicts)
        raise FileExistsError(
            f"Salidas ya existentes (use --overwrite para reemplazarlas): {listed}"
        )


def _run_inference_sample(
    interpreter: tf.lite.Interpreter,
    sample: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    if sample.shape != (EXPECTED_BATCH, EXPECTED_HW[0], EXPECTED_HW[1], 3):
        raise VerificationError(
            f"[{label}] tensor de entrada tiene forma {sample.shape}; "
            f"se esperaba {(EXPECTED_BATCH, EXPECTED_HW[0], EXPECTED_HW[1], 3)}"
        )
    expected_dtype = inp["dtype"]
    if sample.dtype != expected_dtype:
        sample = sample.astype(expected_dtype)
    interpreter.set_tensor(inp["index"], sample)
    interpreter.invoke()
    y = np.copy(interpreter.get_tensor(out["index"]))
    if not np.isfinite(y).all():
        raise VerificationError(f"[{label}] salida no finita: {y!r}")
    if (y < -1e-5).any() or (y > 1.0 + 1e-5).any():
        raise VerificationError(
            f"[{label}] salida fuera de [0,1] (sigmoide); min={y.min()} max={y.max()}"
        )
    return y


def verify_tflite_model(tflite_bytes: bytes) -> None:
    """Valida forma dtype y rangos del grafo TFLite (probabilidad sigmoide)."""
    try:
        interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    except Exception as e:
        raise VerificationError(f"No se pudo crear el interpreter TFLite: {e}") from e

    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    target_in = [EXPECTED_BATCH, EXPECTED_HW[0], EXPECTED_HW[1], 3]
    if list(inp["shape"]) != target_in:
        try:
            interpreter.resize_tensor_input(inp["index"], target_in)
        except ValueError as e:
            raise VerificationError(
                f"No se pudo fijar la entrada a {target_in} "
                f"(forma declarada: {list(inp['shape'])}): {e}"
            ) from e
        interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    ih = tuple(inp["shape"])
    oh = tuple(out["shape"])

    if ih != (EXPECTED_BATCH, EXPECTED_HW[0], EXPECTED_HW[1], 3):
        raise VerificationError(
            f"Forma de entrada tras allocate/resize: {ih}; "
            f"se esperaba {(EXPECTED_BATCH, EXPECTED_HW[0], EXPECTED_HW[1], 3)}"
        )
    if oh != (EXPECTED_BATCH, 1):
        raise VerificationError(f"Forma de salida: {oh}; se esperaba {(EXPECTED_BATCH, 1)}")

    if out["dtype"] != np.float32:
        raise VerificationError(
            f"dtype de salida {out['dtype']}; se esperaba float32 (equivalente a Keras en salida)."
        )

    zeros = np.zeros((EXPECTED_BATCH, EXPECTED_HW[0], EXPECTED_HW[1], 3), dtype=np.float32)
    rng = np.random.default_rng(SEED)
    rand_shape = (EXPECTED_BATCH, EXPECTED_HW[0], EXPECTED_HW[1], 3)
    random_img = rng.uniform(0.0, 1.0, size=rand_shape).astype(np.float32)

    _run_inference_sample(interpreter, zeros, label="zeros")
    _run_inference_sample(interpreter, random_img, label="random")


def _convert_keras_to_tflite(model: keras.Model) -> bytes:
    """Convierte con ``Optimize.DEFAULT``; verificación exige salida float32."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    try:
        return converter.convert()
    except Exception as e:
        raise ConversionError(
            "Falló tf.lite.TFLiteConverter.convert(). Revise compatibilidad de operadores "
            "y versión de TensorFlow."
        ) from e


def _build_metadata() -> dict[str, object]:
    return {
        "model_version": MODEL_VERSION,
        "input_size": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
        "temperature": TEMPERATURE,
        "operational_threshold": OPERATIONAL_THRESHOLD,
        "preprocessing": PREPROCESSING,
        "output_type": "probability",
        "calibration_required": True,
        "thresholding_required": True,
        "raw_output_is_sigmoid_probability": True,
        "temperature_scaling_applied_inside_graph": False,
        "post_processing_order": [
            "Decode/float32 RGB tensor shaped [1,224,224,3]",
            PREPROCESSING,
            "Run TFLite inference → scalar raw_prob ∈ [0,1]",
            "Apply temperature scaling to raw_prob using `temperature` (same formula as backend)",
            "Compare calibrated probability to `operational_threshold` for binary prediction",
        ],
        "notes": (
            "Salida .tflite = probabilidad sigmoide sin calibrar (como raw_probability del API). "
            "Aplicar temperature fuera del grafo. Usar operational_threshold sobre la "
            "probabilidad ya calibrada, no sobre la salida cruda."
        ),
    }


def _print_summary(
    *,
    keras_path: Path,
    out_tflite: Path,
    out_meta: Path,
    model_version: str,
    tflite_bytes_len: int,
) -> None:
    print()
    print("=== Export TensorFlow Lite — resumen ===")
    print(f"  Keras (origen):     {keras_path}")
    print(f"  TFLite (destino):  {out_tflite}  ({tflite_bytes_len} bytes)")
    print(f"  Metadatos:         {out_meta}")
    print(f"  model_version:     {model_version}")
    print("=========================================")
    print()


def main() -> int:
    args = _parse_args()
    keras_path: Path = args.keras_path.resolve()
    if not keras_path.is_file():
        print(f"No existe el modelo: {keras_path}", file=sys.stderr)
        return 1

    try:
        out_tflite = _assert_path_under_models_dir(args.output_tflite, label="--output-tflite")
        out_meta = _assert_path_under_models_dir(args.output_metadata, label="--output-metadata")
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    try:
        _check_outputs_exist([out_tflite, out_meta], overwrite=args.overwrite)
    except FileExistsError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(f"Cargando Keras: {keras_path}")
    try:
        model = keras.models.load_model(keras_path)
    except Exception as e:
        print(f"Error al cargar .keras: {e}", file=sys.stderr)
        return 1

    inp_shape = tuple(model.input_shape)
    if len(inp_shape) != 4 or inp_shape[-1] != 3:
        print(f"Forma de entrada inesperada: {inp_shape}", file=sys.stderr)
        return 1
    h, w = int(inp_shape[1]), int(inp_shape[2])
    if (h, w) != EXPECTED_HW:
        print(
            f"Error: modelo con entrada {h}x{w}; se requiere {EXPECTED_HW[0]}x{EXPECTED_HW[1]}.",
            file=sys.stderr,
        )
        return 1

    try:
        tflite_bytes = _convert_keras_to_tflite(model)
    except ConversionError as e:
        print(f"Conversión TFLite: {e}", file=sys.stderr)
        if e.__cause__ is not None:
            print(f"  Causa: {e.__cause__!r}", file=sys.stderr)
        return 1

    if args.verify:
        try:
            verify_tflite_model(tflite_bytes)
        except VerificationError as e:
            print(f"Verificación TFLite fallida: {e}", file=sys.stderr)
            return 2
    else:
        print("Aviso: verificación omitida (--no-verify).", file=sys.stderr)

    metadata = _build_metadata()

    try:
        out_tflite.parent.mkdir(parents=True, exist_ok=True)
        out_tflite.write_bytes(tflite_bytes)
        out_meta.write_text(json.dumps(metadata, indent=2, sort_keys=False), encoding="utf-8")
    except OSError as e:
        print(f"Error al escribir artefactos: {e}", file=sys.stderr)
        return 1

    _print_summary(
        keras_path=keras_path,
        out_tflite=out_tflite,
        out_meta=out_meta,
        model_version=MODEL_VERSION,
        tflite_bytes_len=len(tflite_bytes),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
