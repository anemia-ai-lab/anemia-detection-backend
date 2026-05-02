#!/usr/bin/env python3
"""
Entrena el baseline MobileNetV2 (transfer learning: cabezal + fine-tuning opcional).

Incluye: ``class_weight`` opcional (por defecto activo), oversampling opcional de positivos en train,
augmentación moderada solo en train, evaluación en test y un informe (Markdown + JSON).

Uso (desde la raíz del repo o desde ``ml/``)::

    cd ml && pip install -r requirements.txt
    python scripts/train.py --demo
    python scripts/train.py --train-dir data/train --test-dir data/test \\
        --metadata-path data_raw/nature/metadata.csv
    python scripts/train.py --oversample-positive \\
        --baseline-experiment-json artifacts/runs/experiment_YYYYMMDDTHHMMSSZ.json

    Fine-tuning parcial del backbone (misma data/augment que arriba), p. ej. frente al mejor run
    oversampling + ``--no-class-weight`` sin fine-tuning previo::

    python scripts/train.py --oversample-positive --no-class-weight \\
        --fine-tune-epochs 10 --fine-tune-learning-rate 1e-5 \\
        --baseline-experiment-json artifacts/runs/experiment_20260420T042800Z.json

    Calibración de probabilidades (``temperature scaling``) sobre un .keras ya entrenado, sin
    modificar pesos: ``python scripts/calibrate_eval.py --experiment-json artifacts/runs/....json``.

Seguimiento opcional en MLflow (backend local ``file:…/ml/mlruns`` por defecto; ver
``baseline/mlflow_logging.py`` para variables de entorno). Si MLflow falla, el entrenamiento y los
informes JSON/Markdown **no** se ven afectados::

    python scripts/train.py … --mlflow
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from baseline.config import (  # noqa: E402
    AUG_BRIGHTNESS_MAX_DELTA,
    AUG_CONTRAST_FACTOR,
    AUG_MAX_ROTATION_FACTOR,
    AUG_ZOOM_RANGE,
    DATA_ROOT,
    DEFAULT_BEST_MODEL_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_TRAIN_DIR,
    EXPERIMENT_MODEL_VERSION,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_FREEZE_UP_TO_LAYER,
    FINE_TUNE_LEARNING_RATE,
    HEAD_EPOCHS,
    HEAD_LEARNING_RATE,
    IMG_SIZE,
    MODEL_DIR,
    RUNS_DIR,
)
from baseline.dataops import (  # noqa: E402
    count_images_in_class_folders,
    count_total_crops,
    count_unique_patients_from_crops,
    load_test_dataset,
    load_train_val_datasets,
    make_demo_datasets,
    read_unique_patient_count_from_metadata,
    write_json,
    write_text,
)
from baseline.evaluation import (  # noqa: E402
    build_threshold_evaluation_results,
    collect_binary_predictions,
)
from baseline.model import (  # noqa: E402
    BACKBONE_LAYER_NAME,
    backbone_partial_unfreeze_counts,
    build_model,
    compile_for_binary,
    set_backbone_trainable,
)
from tensorflow import keras  # noqa: E402


def _software_versions() -> dict[str, str]:
    import tensorflow as tf

    k_ver = getattr(keras, "__version__", None) or getattr(
        getattr(tf, "keras", None), "__version__", "unknown"
    )
    return {"tensorflow": str(tf.__version__), "keras": str(k_ver)}


def _callbacks_configuration(best_checkpoint_path: Path) -> list[dict[str, object]]:
    return [
        {
            "class": "ModelCheckpoint",
            "filepath": str(best_checkpoint_path),
            "monitor": "val_auc",
            "mode": "max",
            "save_best_only": True,
            "verbose": 1,
        },
        {
            "class": "EarlyStopping",
            "monitor": "val_auc",
            "patience": 5,
            "restore_best_weights": True,
            "mode": "max",
            "verbose": 1,
        },
        {
            "class": "ReduceLROnPlateau",
            "monitor": "val_auc",
            "factor": 0.3,
            "patience": 3,
            "mode": "max",
            "verbose": 1,
        },
    ]


def _training_callbacks(best_checkpoint_path: Path) -> list[keras.callbacks.Callback]:
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.3,
            patience=3,
            mode="max",
            verbose=1,
        ),
    ]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenar baseline MobileNetV2 (binario).")
    p.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Carpeta con dos subcarpetas de clase (p. ej. negative/ y positive/).",
    )
    p.add_argument(
        "--test-dir",
        type=Path,
        default=DATA_ROOT / "test",
        help="Carpeta test con la misma estructura que train (solo evaluación final).",
    )
    p.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="CSV Nature (opcional): cuenta única de PATIENT_ID para el informe.",
    )
    p.add_argument(
        "--output-model",
        type=Path,
        default=None,
        help="Ruta del .keras (por defecto artifacts/models/baseline_mobilenetv2.keras).",
    )
    p.add_argument("--head-epochs", type=int, default=HEAD_EPOCHS)
    p.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    p.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=FINE_TUNE_LEARNING_RATE,
        metavar="LR",
        help="LR de Adam en la fase 2 (fine-tuning parcial del backbone; por defecto 1e-5).",
    )
    p.add_argument(
        "--fine-tune-freeze-up-to-layer",
        type=int,
        default=FINE_TUNE_FREEZE_UP_TO_LAYER,
        metavar="N",
        help=(
            "Corte en subcapas del backbone: congela índices < N; N negativo = offset desde el final "
            f"(por defecto {FINE_TUNE_FREEZE_UP_TO_LAYER}, últimas |N| entrenables)."
        ),
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Ignora imágenes en disco y entrena con batches sintéticos (smoke test).",
    )
    p.add_argument("--validation-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-augment",
        action="store_true",
        help="Desactiva augmentación en train (solo depuración).",
    )
    p.add_argument(
        "--no-class-weight",
        action="store_true",
        help=(
            "No pasa `class_weight` a `model.fit` (pesos uniformes implícitos). "
            "Por defecto los pesos balanceados siguen activos."
        ),
    )
    p.add_argument(
        "--oversample-positive",
        action="store_true",
        help=(
            "Duplica aleatoriamente muestras positivas en el train (split interno) hasta ~1:1; "
            "no afecta val/test. Los pesos balanceados se calculan antes del oversampling "
            "(pueden desactivarse en el fit con --no-class-weight)."
        ),
    )
    p.add_argument(
        "--baseline-experiment-json",
        type=Path,
        default=None,
        help=(
            "JSON de un experimento previo para la tabla Δ (AUC, recall @operacional, etc.). "
            "Para evaluar fine-tuning, suele pasarse el mejor JSON *oversampling + --no-class-weight* "
            "sin fase 2 (p. ej. `experiment_20260420T042800Z.json`)."
        ),
    )
    p.add_argument(
        "--mlflow",
        action="store_true",
        help=(
            "Registrar parámetros y métricas en MLflow (URI local ``file:<ml>/mlruns`` por defecto; "
            "sobrescribible con MLFLOW_TRACKING_URI / MLFLOW_EXPERIMENT_NAME)."
        ),
    )
    return p.parse_args()


def _metrics_for_compare(results: dict) -> dict[str, float | None]:
    m05 = results.get("at_threshold_0_5") or {}
    mop = results.get("at_operational_threshold") or results.get("at_youden_optimal") or {}
    return {
        "auc": results.get("auc"),
        "loss": results.get("loss"),
        "recall_at_operational": mop.get("recall"),
        "precision_at_operational": mop.get("precision"),
        "accuracy_at_operational": mop.get("accuracy"),
        "recall_at_0.5": m05.get("recall"),
        "precision_at_0.5": m05.get("precision"),
        "accuracy_at_0.5": m05.get("accuracy"),
    }


def _delta_numeric(
    current: dict[str, float | None],
    baseline: dict[str, float | None],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in current:
        a, b = current.get(k), baseline.get(k)
        if a is None or b is None:
            out[k] = None
            continue
        try:
            out[k] = float(a) - float(b)
        except (TypeError, ValueError):
            out[k] = None
    return out


def _fmt_counts(class_names: list[str], counts: dict[int, int]) -> str:
    parts = []
    for i, name in enumerate(class_names):
        parts.append(f"  clase {i} ({name}): {counts.get(i, 0)}")
    return "\n".join(parts)


def _experiment_markdown(payload: dict) -> str:
    d = payload["dataset_preparation"]
    dist = payload["class_distribution"]
    imb = payload["class_imbalance"]
    cfg = payload["training_configuration"]
    bft = cfg.get("backbone_fine_tuning") if isinstance(cfg.get("backbone_fine_tuning"), dict) else {}
    p1 = bft.get("phase_1_head") if isinstance(bft.get("phase_1_head"), dict) else {}
    p2 = bft.get("phase_2_partial_backbone") if isinstance(bft.get("phase_2_partial_backbone"), dict) else {}
    if bft.get("enabled") and p2:
        fi = p2.get("first_trainable_layer_index")
        if fi is not None:
            idx_note = f"primera subcapa entrenable: índice **{fi}**"
        else:
            idx_note = "**todas** las subcapas del backbone entrenables (`freeze_up_to_layer` = 0)"
        phase2_md = (
            f"- **Fase 2 — fine-tuning parcial:** **{p2.get('epochs')}** épocas, LR **{p2.get('learning_rate')}**; "
            f"subcapas backbone **{p2.get('backbone_sublayers_total')}**: congeladas **{p2.get('backbone_layers_frozen')}**, "
            f"entrenables **{p2.get('backbone_layers_unfrozen')}**; "
            f"`--fine-tune-freeze-up-to-layer` = **{p2.get('freeze_up_to_layer')}** ({idx_note})."
        )
    else:
        phase2_md = "- **Fase 2:** no aplicada (`--fine-tune-epochs` = 0)."
    d_ft_cfg: list[str] = [
        f"- **Fine-tuning parcial del backbone (fase 2):** "
        f"{'**habilitado**' if bft.get('enabled') else '**no**'} "
        f"(`--fine-tune-epochs`: {cfg.get('fine_tune_epochs', 0)}).",
    ]
    if bft.get("enabled") and p2:
        d_ft_cfg.extend(
            [
                f"- **Subcapas MobileNetV2 (fase 2):** total **{p2.get('backbone_sublayers_total')}**; "
                f"congeladas **{p2.get('backbone_layers_frozen')}**; entrenables **{p2.get('backbone_layers_unfrozen')}**.",
                f"- **Corte de congelación:** `--fine-tune-freeze-up-to-layer` = **{p2.get('freeze_up_to_layer')}** "
                f"(LR fase 2: **{p2.get('learning_rate')}**).",
            ],
        )
    res = payload["results"]
    sw = payload.get("software_versions") or {}
    cb = cfg.get("callbacks_configuration") or []
    thr = res.get("thresholds_used") or {}
    sct = dist.get("split_class_counts") or {}

    def _f(v: object, *, nd: int = 6) -> str:
        if v is None:
            return "N/A"
        return f"{float(v):.{nd}f}"

    m05 = res.get("at_threshold_0_5") or {}
    mop = res.get("at_operational_threshold") or {}
    my = res.get("at_youden_optimal") or {}
    ci = res.get("clinical_interpretation") or {}

    def _split_line(label: str, key: str) -> str:
        block = sct.get(key) or {}
        c0 = block.get("0", "N/A")
        c1 = block.get("1", "N/A")
        return f"- **{label}** — clase 0 ({dist.get('class_name_0', '?')}): {c0}; clase 1 ({dist.get('class_name_1', '?')}): {c1}"

    cb_lines = "\n".join(f"  - `{c.get('class')}`: {c}" for c in cb) if cb else "  - (no registrado)"
    osb = payload.get("oversampling") or {}
    bc = payload.get("baseline_comparison") or {}

    def _oversampling_md_block() -> list[str]:
        if not osb.get("oversampling_requested"):
            return []
        c0n = dist.get("class_name_0", "clase_0")
        c1n = dist.get("class_name_1", "clase_1")
        bef = osb.get("before_train_by_class") or {}
        aft = osb.get("after_train_by_class") or {}
        blk = [
            "**Oversampling (solo subconjunto train del `fit`, no val/test):**",
            "- **Solicitado (`--oversample-positive`):** sí.",
            f"- **Aplicado a los datos:** {'**sí**' if osb.get('applied') else '**no**'}",
        ]
        if not osb.get("applied") and osb.get("reason"):
            blk.append(f"- **Motivo si no hubo duplicación:** `{osb.get('reason')}`")
        blk.extend(
            [
                f"- **Distribución ANTES (conteos por clase, {c0n}/{c1n}):** "
                f"n₀={bef.get('0', 'N/A')}, n₁={bef.get('1', 'N/A')}",
                f"- **Distribución DESPUÉS:** n₀={aft.get('0', 'N/A')}, n₁={aft.get('1', 'N/A')}",
                f"- **Duplicados positivos añadidos:** {osb.get('duplicates_added', 0)}",
            ]
        )
        if osb.get("strategy"):
            blk.append(f"- **Método:** {osb['strategy']}")
        if imb.get("class_weight_used"):
            blk.append(
                "- **Nota:** los conteos **antes** del oversampling documentan el split; "
                "`class_weight` activo en `fit` se deriva de esos conteos previos al oversampling."
            )
        else:
            blk.append(
                "- **Nota:** `class_weight` estaba **desactivado** en `fit` (`--no-class-weight`); "
                "los conteos antes/después del oversampling siguen documentados para trazabilidad."
            )
        return blk + [""]

    def _baseline_md_block() -> list[str]:
        if not bc.get("baseline_experiment_path"):
            return []
        if bc.get("error"):
            return [
                "### Comparación con baseline (`--baseline-experiment-json`)",
                "",
                f"- **Ruta solicitada:** `{bc.get('baseline_experiment_path')}`",
                f"- **Error:** {bc.get('error')}",
                "",
            ]
        cur = bc.get("metrics_current") or {}
        base = bc.get("metrics_baseline") or {}
        dlt = bc.get("delta_current_minus_baseline") or {}
        bid = bc.get("baseline_run_id") or "N/A"
        snap = bc.get("baseline_training_snapshot") or {}
        lines_b = [
            "### Comparación con baseline (`--baseline-experiment-json`)",
            "",
            f"- **JSON baseline:** `{bc.get('baseline_experiment_path')}`",
            f"- **run_id baseline:** `{bid}`",
            "",
        ]
        if snap:
            b_oe = snap.get("backbone_fine_tuning_enabled")
            lines_b.extend(
                [
                    "- **Perfil del run baseline (desde JSON):**",
                    f"  - Épocas **cabezal** / **fine-tuning backbone:** {snap.get('head_epochs')} / "
                    f"{snap.get('fine_tune_epochs')}",
                    f"  - Fine-tuning del backbone en baseline: **{'sí' if b_oe else 'no'}**",
                    f"  - Oversampling (`--oversample-positive` / aplicado en datos): "
                    f"{snap.get('oversample_positive_cli')} / {snap.get('oversampling_used')}",
                    f"  - `class_weight` en `fit` (baseline): **{'sí' if snap.get('class_weight_enabled_in_fit') else 'no'}** "
                    f"(`no_class_weight` en JSON: {snap.get('no_class_weight')})",
                    "",
                ],
            )
        lines_b.extend(
            [
                "| Métrica | Baseline | Este run | Δ (este − baseline) |",
                "|---------|----------|----------|------------------------|",
            ],
        )
        rows = [
            ("AUC", "auc"),
            ("Loss", "loss"),
            ("Recall @operacional (Youden)", "recall_at_operational"),
            ("Precision @operacional (Youden)", "precision_at_operational"),
            ("Accuracy @operacional", "accuracy_at_operational"),
            ("Recall @0.5 (ref.)", "recall_at_0.5"),
            ("Precision @0.5 (ref.)", "precision_at_0.5"),
            ("Accuracy @0.5 (ref.)", "accuracy_at_0.5"),
        ]
        for label, key in rows:
            bv, cv, dv = base.get(key), cur.get(key), dlt.get(key)

            def cell(x: object) -> str:
                if x is None:
                    return "N/A"
                return f"{float(x):.6f}"

            d_cell = "N/A" if dv is None else cell(dv)
            lines_b.append(f"| {label} | {cell(bv)} | {cell(cv)} | {d_cell} |")
        lines_b.append("")
        lines_b.append(
            "_Δ > 0 en AUC/recall suele indicar mejora respecto al baseline; en loss depende del signo "
            "(menor loss es mejor)._",
        )
        lines_b.append("")
        lines_b.append(
            "_En experimentos de fine-tuning, suele cargarse como baseline el JSON del mejor run "
            "«oversampling + sin `class_weight`» **sin** segunda fase de backbone, para ver si la "
            "fase 2 mejora AUC y recall @operacional._",
        )
        lines_b.append("")
        return lines_b

    extra_train_before: list[str] = []
    if sct.get("train_fit_before_oversampling"):
        extra_train_before = [
            _split_line(
                "Entrenamiento antes de oversampling (conteos usados para `class_weight`)",
                "train_fit_before_oversampling",
            ),
            "",
        ]

    lines = [
        "# Informe de experimento (baseline CNN)",
        "",
        f"**Run ID:** `{payload['run_id']}`  ",
        f"**Marca de tiempo (UTC):** `{payload['timestamp_utc']}`  ",
        f"**model_version:** `{payload.get('model_version', 'N/A')}`  ",
        f"**Semilla aleatoria (reproducibilidad):** `{payload.get('random_seed', 'N/A')}`",
        "",
        "**Versiones de software:**",
        f"- TensorFlow: `{sw.get('tensorflow', 'N/A')}`",
        f"- Keras: `{sw.get('keras', 'N/A')}`",
        "",
        "**Callbacks (configuración):**",
        cb_lines,
        "",
        "**Umbrales de decisión (evaluación en test):**",
        f"- **Umbral teórico de referencia (sigmoid):** **{thr.get('theoretical_default_threshold', thr.get('fixed_decision_threshold', 0.5))}** — no se usa como corte operacional.",
        f"- **Umbral operacional (predicciones / interpretación):** **{_f(thr.get('operational_threshold', thr.get('roc_youden_optimal_threshold')))}** "
        f"(ROC, máximo índice de Youden; J = {_f(thr.get('youden_j_at_operational_threshold', thr.get('youden_j_max')))}).",
        f"- {thr.get('operational_for_predictions_note', '')}",
        "",
        "**Muestreo / pesos (resumen del experimento):**",
        f"- **`class_weight` en `model.fit`:** {'**habilitado**' if imb.get('class_weight_used') else '**desactivado**'} "
        f"({'`--no-class-weight`' if not imb.get('class_weight_used') else 'por defecto'}).",
        f"- **Oversampling en train (datos):** {'**sí**' if imb.get('oversampling_used') else '**no**'} "
        f"(solicitado por CLI: {'sí' if imb.get('oversampling_requested') else 'no'}).",
        f"- {imb.get('baseline_sampling_note', '')}",
        "",
        "**Entrenamiento en dos fases (MobileNetV2):**",
        f"- **Fine-tuning parcial del backbone (fase 2):** "
        f"{'**habilitado**' if bft.get('enabled') else '**no**'} "
        f"(épocas fase 2 según CLI: {cfg.get('fine_tune_epochs', 0)}).",
        f"- **Fase 1 — cabezal (backbone congelado):** **{cfg.get('head_epochs')}** épocas, "
        f"LR **{p1.get('learning_rate', cfg.get('head_learning_rate'))}**; "
        f"subcapas en `mobilenet_backbone`: **{p1.get('backbone_sublayers_total', 'N/A')}** "
        "(ImageNet **no** se actualiza en esta fase).",
        phase2_md,
        "",
        "**Conteos por clase (splits):**",
        _split_line("Entrenamiento (subconjunto fit, efectivo para `fit`)", "train_fit"),
        _split_line("Validación (subconjunto fit)", "validation_fit"),
        _split_line("Test (carpeta `data/test`)", "test_folder"),
        "",
        *extra_train_before,
        *_oversampling_md_block(),
        "## a) Preparación del dataset",
        "",
        d["description"],
        "",
        f"- **Pacientes (estudio Nature, únicos):** {d['total_patients']}",
        f"  - Fuente del conteo: {d['patient_count_source']}",
        f"- **Total de muestras (crops) en disco:** {d['total_crops_all_splits']}",
        f"  - Train (carpeta): {d['crops_train_folder']}",
        f"  - Test (carpeta): {d['crops_test_folder']}",
        f"- **Estrategia train/validation:** {d['train_val_split_strategy']}",
        f"- **Estrategia train/test (alto nivel):** {d['train_test_split_strategy']}",
        "",
        "## b) Distribución de clases (crops en carpetas)",
        "",
        f"- **Negative:** {dist['negative_total']}",
        f"- **Positive:** {dist['positive_total']}",
        "",
        "## c) Desbalance de clases",
        "",
        imb["explanation"],
        "",
        f"- **`class_weight` en `model.fit`:** {'**habilitado**' if imb.get('class_weight_used') else '**desactivado**'} "
        f"({'`--no-class-weight`' if imb.get('class_weight_disabled_by_cli') else 'por defecto'}).",
        f"- **Oversampling en train (datos):** {'**sí**' if imb.get('oversampling_used') else '**no**'} "
        f"(solicitado CLI: {'sí' if imb.get('oversampling_requested') else 'no'}).",
        "",
        f"{imb.get('baseline_sampling_note', '')}",
        "",
        *(
            [
                (
                    "Conteos en el subconjunto **train** tras el split (originales; base para "
                    "`class_weight` **si** está habilitado en `fit`, **antes** de oversampling si aplica):"
                ),
                "",
                f"- Clase 0 ({dist['class_name_0']}): {imb['train_subset_counts']['0']}",
                f"- Clase 1 ({dist['class_name_1']}): {imb['train_subset_counts']['1']}",
                "",
                "Pesos **aplicados** en `model.fit(..., class_weight=...)`:",
                "",
                f"- Peso clase 0: **{imb['computed_class_weights']['0']}**",
                f"- Peso clase 1: **{imb['computed_class_weights']['1']}**",
                "",
            ]
            if imb.get("class_weight_used")
            else [
                (
                    "Conteos en el subconjunto **train** tras el split (originales; **antes** de oversampling "
                    "si aplica). Con `--no-class-weight`, no se usaron para ponderar el loss en `fit`, "
                    "pero se documentan para trazabilidad:"
                ),
                "",
                f"- Clase 0 ({dist['class_name_0']}): {imb['train_subset_counts']['0']}",
                f"- Clase 1 ({dist['class_name_1']}): {imb['train_subset_counts']['1']}",
                "",
                (
                    "**`class_weight` no aplicado en `fit`** (`--no-class-weight`). "
                    "Pesos balanceados de **referencia** (lo que se habría usado por defecto):"
                ),
                "",
                f"- Referencia clase 0: **{imb['computed_class_weights']['0']}**",
                f"- Referencia clase 1: **{imb['computed_class_weights']['1']}**",
                "",
            ]
        ),
        "## d) Configuración de entrenamiento",
        "",
        f"- **Modelo:** {cfg['model_name']}",
        f"- **model_version (artefacto):** `{payload.get('model_version', 'N/A')}`",
        f"- **Tamaño de entrada:** {cfg['input_size']}",
        f"- **Fase 1 — épocas cabezal (solo cabezal):** {cfg['head_epochs']}",
        f"- **Fase 2 — épocas fine-tuning backbone:** {cfg['fine_tune_epochs']}",
        f"- **Learning rate cabezal (fase 1):** {cfg['head_learning_rate']}",
        f"- **Learning rate fine-tuning (fase 2):** {cfg['fine_tune_learning_rate']}",
        f"- **Augmentación (solo train):** {cfg['augmentation_summary']}",
        f"- **Semilla:** {cfg['seed']}",
        f"- **validation_split (interno):** {cfg['validation_split']}",
        f"- **`class_weight` en entrenamiento:** "
        f"{'habilitado (por defecto)' if not cfg.get('no_class_weight') else 'desactivado (`--no-class-weight`)'}",
        f"- **Oversampling (`--oversample-positive`):** "
        f"{'solicitado' if cfg.get('oversample_positive') else 'no solicitado'}",
        *d_ft_cfg,
        "",
        "## e) Resultados",
        "",
        f"- **Conjunto de evaluación:** {res['evaluation_dataset']}",
        "",
        "### Interpretación clínica de umbrales",
        "",
    ]
    if ci:
        lines.extend(
            [
                f"- **Por qué 0.5 no es el umbral operacional por defecto:** {ci.get('why_threshold_0_5_not_default', '')}",
                f"- **Por qué se prioriza el recall (sensibilidad):** {ci.get('why_prioritize_recall', '')}",
                f"- **Uso del umbral operacional:** {ci.get('operational_threshold_use', '')}",
                "",
            ],
        )
    else:
        lines.extend(["- *(Sin evaluación en test, p. ej. modo `--demo`.)*", "",])

    lines.extend(
        [
        "| Métrica global | Valor |",
        "|----------------|-------|",
        f"| Loss (test) | {_f(res.get('loss'))} |",
        f"| **AUC (ROC, independiente del umbral)** | {_f(res.get('auc'))} |",
        "",
        "### Métricas principales (umbral operacional = ROC-Youden)",
        "",
        (
            "Estas son las métricas **destacadas** para el contexto de tesis / uso clínico documentado; "
            "corresponden al corte **operacional** (máximo J en ROC sobre test)."
        ),
        "",
        f"- **Umbral operacional τ:** `{_f(mop.get('threshold', my.get('optimal_threshold')), nd=6)}`  ",
        f"- **Índice de Youden J:** {_f(mop.get('youden_j', my.get('youden_j')))}",
        "",
        "| **Precision** | **Recall (sensibilidad)** | Accuracy |",
        "|---------------|---------------------------|----------|",
        f"| **{_f(mop.get('precision', my.get('precision')))}** | **{_f(mop.get('recall', my.get('recall')))}** | {_f(mop.get('accuracy', my.get('accuracy')))} |",
        "",
        "### Referencia: umbral teórico 0.5 (sigmoid)",
        "",
        "Solo comparación bibliográfica; **no** se recomienda como decisión principal con datos desbalanceados.",
        "",
        "| Accuracy | Precision | Recall (sensibilidad) |",
        "|----------|-----------|------------------------|",
        f"| {_f(m05.get('accuracy'))} | {_f(m05.get('precision'))} | {_f(m05.get('recall'))} |",
        "",
        *_baseline_md_block(),
        "## f) Artefacto del modelo",
        "",
        f"- **Ruta .keras:** `{res['model_path']}`",
        "",
        ],
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    run_started = datetime.now(timezone.utc)
    run_id = run_started.strftime("%Y%m%dT%H%M%SZ")
    timestamp_utc_iso = run_started.isoformat()
    out_path = (args.output_model or (MODEL_DIR / DEFAULT_MODEL_NAME)).expanduser().resolve()
    best_checkpoint_path = (MODEL_DIR / DEFAULT_BEST_MODEL_NAME).expanduser().resolve()
    train_dir = args.train_dir.expanduser().resolve()
    test_dir = args.test_dir.expanduser().resolve()
    metadata_path = args.metadata_path.expanduser().resolve() if args.metadata_path else None

    augment_train = not args.no_augment

    oversampling_info: dict[str, object] = {
        "applied": False,
        "reason": "not_loaded",
        "before_train_by_class": {"0": 0, "1": 0},
        "after_train_by_class": {"0": 0, "1": 0},
        "duplicates_added": 0,
    }

    if args.demo:
        train_ds, val_ds = make_demo_datasets()
        class_names = ["negative", "positive"]
        train_counts = {0: 1, 1: 1}
        val_counts = {0: 0, 1: 0}
        class_weight = {0: 1.0, 1: 1.0}
        oversampling_info = {
            "applied": False,
            "reason": "demo_mode",
            "before_train_by_class": {"0": 1, "1": 1},
            "after_train_by_class": {"0": 1, "1": 1},
            "duplicates_added": 0,
        }
        print(
            "[demo] pesos neutros {0: 1.0, 1: 1.0} para referencia; en fit: "
            f"{'sin class_weight' if args.no_class_weight else 'con class_weight'}.",
        )
    else:
        if not train_dir.is_dir():
            raise SystemExit(f"No existe --train-dir: {train_dir}")
        (
            train_ds,
            val_ds,
            train_counts,
            val_counts,
            class_weight,
            class_names,
            oversampling_info,
        ) = load_train_val_datasets(
            train_dir,
            validation_split=args.validation_split,
            seed=args.seed,
            augment_train=augment_train,
            oversample_positive=args.oversample_positive,
        )

    class_weight_enabled_in_fit = not args.no_class_weight
    class_weight_for_fit = class_weight if class_weight_enabled_in_fit else None

    print("--- Conteos (subconjunto train tras split) y class_weight ---")
    print(_fmt_counts(class_names, train_counts))
    print(f"Pesos balanceados (referencia / Keras): {class_weight}")
    print(
        "class_weight en model.fit: "
        f"{'sí (habilitado)' if class_weight_enabled_in_fit else 'no (--no-class-weight)'}",
    )
    if oversampling_info.get("applied"):
        bef = oversampling_info.get("before_train_by_class") or {}
        aft = oversampling_info.get("after_train_by_class") or {}
        print(
            f"--- Oversampling positivos (train fit): antes n₀={bef.get('0')}, n₁={bef.get('1')} → "
            f"después n₀={aft.get('0')}, n₁={aft.get('1')} (+{oversampling_info.get('duplicates_added', 0)} duplicados) ---",
        )

    model = build_model(backbone_trainable=False)
    n_bb = len(model.get_layer(BACKBONE_LAYER_NAME).layers)
    backbone_ft_meta: dict[str, object] = {
        "enabled": bool(args.fine_tune_epochs > 0),
        "phase_1_head": {
            "epochs": int(args.head_epochs),
            "learning_rate": float(HEAD_LEARNING_RATE),
            "backbone_weights_trainable": False,
            "backbone_sublayers_total": n_bb,
        },
        "phase_2_partial_backbone": None,
    }
    if args.fine_tune_epochs > 0:
        fr, un, co = backbone_partial_unfreeze_counts(
            n_bb,
            int(args.fine_tune_freeze_up_to_layer),
        )
        backbone_ft_meta["phase_2_partial_backbone"] = {
            "epochs": int(args.fine_tune_epochs),
            "learning_rate": float(args.fine_tune_learning_rate),
            "freeze_up_to_layer": int(args.fine_tune_freeze_up_to_layer),
            "backbone_sublayers_total": n_bb,
            "backbone_layers_frozen": int(fr),
            "backbone_layers_unfrozen": int(un),
            "first_trainable_layer_index": int(co) if co is not None else None,
        }

    compile_for_binary(model, HEAD_LEARNING_RATE)

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        verbose=1,
        class_weight=class_weight_for_fit,
        callbacks=_training_callbacks(best_checkpoint_path),
    )

    history_ft = None
    if args.fine_tune_epochs > 0:
        p2 = backbone_ft_meta["phase_2_partial_backbone"] or {}
        print(
            "--- Fase 2: fine-tuning parcial del backbone ---\n"
            f"  LR={args.fine_tune_learning_rate}, freeze_up_to_layer={args.fine_tune_freeze_up_to_layer}\n"
            f"  Subcapas backbone: {n_bb}; congeladas={p2.get('backbone_layers_frozen')}; "
            f"entrenables={p2.get('backbone_layers_unfrozen')}",
        )
        set_backbone_trainable(model, int(args.fine_tune_freeze_up_to_layer))
        compile_for_binary(model, float(args.fine_tune_learning_rate))
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            verbose=1,
            class_weight=class_weight_for_fit,
            callbacks=_training_callbacks(best_checkpoint_path),
        )

    model.save(out_path)

    test_results: dict = {}
    eval_dataset_label = "No evaluado (--demo)"
    if not args.demo:
        if not test_dir.is_dir():
            raise SystemExit(f"No existe --test-dir: {test_dir}")
        test_ds_eval = load_test_dataset(test_dir)
        raw_eval = model.evaluate(test_ds_eval, return_dict=True, verbose=1)
        loss_v = float(raw_eval["loss"])
        auc_v = float(raw_eval["auc"])

        test_ds_pred = load_test_dataset(test_dir)
        y_true, y_prob = collect_binary_predictions(model, test_ds_pred)
        test_results = build_threshold_evaluation_results(
            loss=loss_v,
            auc_val=auc_v,
            y_true=y_true,
            y_prob=y_prob,
        )
        eval_dataset_label = f"Carpeta test independiente: {test_dir}"

    print("\n--- Métricas finales (test) ---")
    if test_results:
        print(f"  loss: {test_results['loss']:.6f}")
        print(f"  auc (ROC, sin umbral): {test_results['auc']:.6f}")
        mop = test_results.get("at_operational_threshold") or {}
        print(
            "  [operacional / predicciones] τ={:.6f} — precision: {:.6f}, recall: {:.6f}, accuracy: {:.6f}".format(
                float(mop.get("threshold", 0.0)),
                float(mop.get("precision", 0.0)),
                float(mop.get("recall", 0.0)),
                float(mop.get("accuracy", 0.0)),
            ),
        )
        m5 = test_results["at_threshold_0_5"]
        print(
            "  [referencia 0.5] — precision: {:.6f}, recall: {:.6f}, accuracy: {:.6f}".format(
                m5["precision"], m5["recall"], m5["accuracy"],
            ),
        )
    else:
        print("  (omitido en modo --demo)")

    baseline_path = (
        args.baseline_experiment_json.expanduser().resolve()
        if args.baseline_experiment_json
        else None
    )
    baseline_comparison: dict | None = None
    if baseline_path and baseline_path.is_file() and test_results:
        base_exp = json.loads(baseline_path.read_text(encoding="utf-8"))
        br = base_exp.get("results") or {}
        btc = base_exp.get("training_configuration") or {}
        imb_b = base_exp.get("class_imbalance") or {}
        cur_sl = _metrics_for_compare(test_results)
        base_sl = _metrics_for_compare(br)
        bft_b = btc.get("backbone_fine_tuning") if isinstance(btc.get("backbone_fine_tuning"), dict) else {}
        baseline_comparison = {
            "baseline_experiment_path": str(baseline_path),
            "baseline_run_id": base_exp.get("run_id"),
            "metrics_baseline": base_sl,
            "metrics_current": cur_sl,
            "delta_current_minus_baseline": _delta_numeric(cur_sl, base_sl),
            "baseline_training_snapshot": {
                "head_epochs": btc.get("head_epochs"),
                "fine_tune_epochs": btc.get("fine_tune_epochs"),
                "backbone_fine_tuning_enabled": bool(
                    (btc.get("fine_tune_epochs") or 0) > 0
                    or (isinstance(bft_b, dict) and bft_b.get("enabled")),
                ),
                "oversample_positive_cli": btc.get("oversample_positive"),
                "oversampling_used": imb_b.get("oversampling_used"),
                "no_class_weight": btc.get("no_class_weight"),
                "class_weight_enabled_in_fit": imb_b.get("class_weight_enabled_in_fit"),
            },
        }
    elif args.baseline_experiment_json:
        baseline_comparison = {
            "baseline_experiment_path": str(args.baseline_experiment_json.expanduser()),
            "error": (
                "No se pudo leer el JSON baseline o no hay métricas de test en este run "
                "(p. ej. modo --demo)."
            ),
        }

    meta_patients = read_unique_patient_count_from_metadata(metadata_path)
    patient_source = "metadata.csv (PATIENT_ID únicos)"
    if meta_patients is None:
        meta_patients = count_unique_patients_from_crops(train_dir, test_dir)
        patient_source = "heurística desde nombres de archivo de crops (sin metadata)"

    train_folder_counts = count_images_in_class_folders(train_dir) if train_dir.is_dir() else {}
    test_folder_counts = count_images_in_class_folders(test_dir) if test_dir.is_dir() else {}
    neg_name = class_names[0] if len(class_names) > 0 else "negative"
    pos_name = class_names[1] if len(class_names) > 1 else "positive"
    neg_total = int(train_folder_counts.get(neg_name, 0) + test_folder_counts.get(neg_name, 0))
    pos_total = int(train_folder_counts.get(pos_name, 0) + test_folder_counts.get(pos_name, 0))

    crops_train = sum(train_folder_counts.values()) if train_folder_counts else 0
    crops_test = sum(test_folder_counts.values()) if test_folder_counts else 0

    aug_summary = "ninguna (--no-augment o demo)" if (args.demo or not augment_train) else (
        f"RandomRotation (±≤15°, factor={AUG_MAX_ROTATION_FACTOR:.4f} respecto a 2π), "
        f"RandomZoom {AUG_ZOOM_RANGE}, RandomBrightness (max_delta={AUG_BRIGHTNESS_MAX_DELTA}), "
        f"RandomContrast (factor={AUG_CONTRAST_FACTOR}); solo en train, sin augment en val/test."
    )

    cw_serializable = {str(k): float(v) for k, v in class_weight.items()}
    train_counts_serial = {str(k): int(v) for k, v in train_counts.items()}
    val_counts_serial = {str(k): int(val_counts.get(k, 0)) for k in (0, 1)}
    train_fit_effective = dict(oversampling_info.get("after_train_by_class") or train_counts_serial)
    split_class_counts: dict[str, object] = {
        "train_fit": train_fit_effective,
        "validation_fit": dict(val_counts_serial),
        "test_folder": {
            "0": int(test_folder_counts.get(neg_name, 0)),
            "1": int(test_folder_counts.get(pos_name, 0)),
        },
    }
    if oversampling_info.get("applied"):
        split_class_counts["train_fit_before_oversampling"] = dict(
            oversampling_info.get("before_train_by_class") or {},
        )

    cw_on = class_weight_enabled_in_fit
    os_on = bool(oversampling_info.get("applied"))

    if os_on and cw_on:
        sampling_note = (
            "Oversampling de positivos en train del `fit` (~1:1) y `class_weight` **habilitado** "
            "(derivado de conteos **previos** al oversampling)."
        )
    elif os_on and not cw_on:
        sampling_note = (
            "Oversampling de positivos en train del `fit` (~1:1); **`class_weight` desactivado** "
            "en `fit` (`--no-class-weight`)."
        )
    elif args.oversample_positive and not os_on:
        sampling_note = (
            "`--oversample-positive` activo sin duplicación efectiva "
            f"(motivo: `{oversampling_info.get('reason', 'desconocido')}`). "
            f"`class_weight` en fit: {'habilitado' if cw_on else 'desactivado'}."
        )
    elif not os_on and cw_on:
        sampling_note = "Sin oversampling; `class_weight` **habilitado** en `model.fit`."
    else:
        sampling_note = "Sin oversampling; `class_weight` **desactivado** en `fit` (`--no-class-weight`)."

    fit_phases_wording = (
        "las fases de cabezal y fine-tuning del backbone"
        if args.fine_tune_epochs > 0
        else "la fase de cabezal (backbone congelado)"
    )
    imbalance_explanation = (
        "Se calcularon pesos balanceados n/(K·n_c) con K=2 a partir de los conteos del "
        "subconjunto de entrenamiento tras el split train/val, y se pasaron a "
        f"`model.fit(..., class_weight=...)` en {fit_phases_wording}."
        if cw_on
        else (
            "Se calcularon pesos balanceados n/(K·n_c) con K=2 a partir de los conteos del "
            "subconjunto train tras el split (**solo referencia**). **No** se pasaron a "
            "`model.fit` (`--no-class-weight`); el entrenamiento usó pesos de muestra uniformes "
            "implícitos en Keras."
        )
    )

    oversampling_report = dict(oversampling_info)
    oversampling_report["oversampling_requested"] = bool(args.oversample_positive)

    experiment: dict = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc_iso,
        "model_version": EXPERIMENT_MODEL_VERSION,
        "random_seed": args.seed,
        "software_versions": _software_versions(),
        "dataset_preparation": {
            "description": (
                "Imágenes recortadas a 224×224 a partir del dataset Nature, usando las regiones "
                "definidas en la columna NAIL_BOUNDING_BOXES del CSV de metadatos (script "
                "`ml/scripts/prepare_nature_dataset.py`). Cada PNG corresponde a un crop de uña."
            ),
            "total_patients": meta_patients,
            "patient_count_source": patient_source,
            "total_crops_all_splits": count_total_crops(train_dir, test_dir) if not args.demo else None,
            "crops_train_folder": crops_train,
            "crops_test_folder": crops_test,
            "train_val_split_strategy": (
                "Partición estratificada aleatoria sobre las imágenes de `train/` "
                f"(validation_split={args.validation_split}, semilla={args.seed}); "
                "las mismas carpetas de clase que usa Keras, orden alfabético de subcarpetas."
            ),
            "train_test_split_strategy": (
                "Train vs test a nivel de **paciente** en la preparación del dataset: "
                "`prepare_nature_dataset.py` asigna todos los crops de un PATIENT_ID solo a train "
                "o solo a test (80/20 por defecto). Las carpetas `data/train` y `data/test` "
                "reflejan ese reparto; la validación durante `fit` es un subconjunto interno de "
                "`data/train`."
            ),
        },
        "class_distribution": {
            "class_name_0": neg_name,
            "class_name_1": pos_name,
            "negative_total": neg_total,
            "positive_total": pos_total,
            "per_folder_train": train_folder_counts,
            "per_folder_test": test_folder_counts,
            "split_class_counts": split_class_counts,
        },
        "class_imbalance": {
            "computed_class_weights": cw_serializable,
            "train_subset_counts": train_counts_serial,
            "class_weight_used": class_weight_enabled_in_fit,
            "class_weight_enabled_in_fit": class_weight_enabled_in_fit,
            "class_weight_disabled_by_cli": bool(args.no_class_weight),
            "oversampling_used": bool(oversampling_info.get("applied")),
            "oversampling_requested": bool(args.oversample_positive),
            "baseline_sampling_note": sampling_note,
            "explanation": imbalance_explanation,
        },
        "training_configuration": {
            "model_name": "MobileNetV2 (ImageNet) + GAP + Dropout + Dense(1, sigmoid)",
            "input_size": list(IMG_SIZE),
            "head_epochs": args.head_epochs,
            "fine_tune_epochs": args.fine_tune_epochs,
            "head_learning_rate": HEAD_LEARNING_RATE,
            "fine_tune_learning_rate": float(args.fine_tune_learning_rate),
            "fine_tune_freeze_up_to_layer": int(args.fine_tune_freeze_up_to_layer),
            "backbone_fine_tuning": backbone_ft_meta,
            "augmentation_summary": aug_summary,
            "seed": args.seed,
            "validation_split": args.validation_split,
            "demo": args.demo,
            "oversample_positive": bool(args.oversample_positive),
            "no_class_weight": bool(args.no_class_weight),
            "baseline_experiment_json": str(baseline_path) if args.baseline_experiment_json else None,
            "callbacks_configuration": _callbacks_configuration(best_checkpoint_path),
        },
        "oversampling": oversampling_report,
        "baseline_comparison": baseline_comparison,
        "results": {
            "evaluation_dataset": eval_dataset_label,
            "model_path": str(out_path),
            **(
                test_results
                if test_results
                else {
                    "loss": None,
                    "auc": None,
                    "auc_note": "AUC de Keras (ROC); independiente del umbral de decisión.",
                    "clinical_interpretation": None,
                    "thresholds_used": None,
                    "at_operational_threshold": None,
                    "at_threshold_0_5": None,
                    "at_youden_optimal": None,
                }
            ),
        },
        "keras_histories": {
            "head": {k: [float(x) for x in v] for k, v in history_head.history.items()},
            "fine_tune": (
                {k: [float(x) for x in v] for k, v in history_ft.history.items()}
                if history_ft
                else None
            ),
        },
    }

    json_path = RUNS_DIR / f"experiment_{run_id}.json"
    md_path = RUNS_DIR / f"experiment_{run_id}.md"
    write_json(json_path, experiment)
    write_text(md_path, _experiment_markdown(experiment))

    mlflow_ok = False
    if args.mlflow:
        from baseline.mlflow_logging import safe_log_train_experiment

        mlflow_ok = safe_log_train_experiment(
            experiment,
            ml_root=_ML_ROOT,
            report_json=json_path,
            report_md=md_path,
            model_path=out_path,
        )

    print(f"\nModelo guardado en: {out_path}")
    print(f"Informe JSON: {json_path}")
    print(f"Informe Markdown: {md_path}")
    if args.mlflow and mlflow_ok:
        print("MLflow: run registrado (ver ml/mlruns o MLFLOW_TRACKING_URI).")


if __name__ == "__main__":
    main()
