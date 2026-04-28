#!/usr/bin/env python3
"""
Calibración post-hoc por *temperature scaling* sobre un modelo .keras ya entrenado.

- Ajusta ``T`` minimizando la NLL (BCE media) en el **mismo subconjunto de validación**
  estratificado que usa ``train.py`` (``validation_split`` + ``seed``).
- Evalúa en **test** probabilidades sin calibrar vs calibradas (sin reentrenar el CNN).

Uso (desde ``ml/``)::

    cd ml && pip install -r requirements.txt
    python scripts/calibrate_eval.py \\
        --model-path artifacts/models/baseline_mobilenetv2.keras \\
        --experiment-json artifacts/runs/experiment_20260420T043804Z.json

Si no se pasa ``--experiment-json``, usar ``--train-dir``, ``--validation-split`` y ``--seed``
coherentes con el entrenamiento original.

Opcional — MLflow (store local por defecto; mismo directorio ``ml/mlruns`` que ``train.py`` si no se
configura ``MLFLOW_TRACKING_URI``). Un fallo en MLflow **no** impide generar los informes ni alterar
métricas::

    python scripts/calibrate_eval.py … --mlflow
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

from baseline.calibration import (  # noqa: E402
    apply_temperature_scaling,
    auc_roc_keras,
    enrich_binary_eval_with_calibration_metrics,
    fit_temperature_scaling_on_probabilities,
    mean_binary_cross_entropy,
)
from baseline.config import (  # noqa: E402
    DEFAULT_TRAIN_DIR,
    DEFAULT_TEST_DIR,
    DEFAULT_MODEL_NAME,
    MODEL_DIR,
    RUNS_DIR,
    SEED,
)
from baseline.dataops import (  # noqa: E402
    load_test_dataset,
    load_validation_dataset,
    write_json,
    write_text,
)
from baseline.evaluation import (  # noqa: E402
    build_threshold_evaluation_results,
    collect_binary_predictions,
)
from tensorflow import keras  # noqa: E402


def _software_versions() -> dict[str, str]:
    import tensorflow as tf

    k_ver = getattr(keras, "__version__", None) or getattr(
        getattr(tf, "keras", None), "__version__", "unknown",
    )
    return {"tensorflow": str(tf.__version__), "keras": str(k_ver)}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Temperature scaling: calibrar probabilidades de un modelo ya entrenado.",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_DIR / DEFAULT_MODEL_NAME,
        help="Pesos .keras del modelo entrenado (no se modifican).",
    )
    p.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    p.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument(
        "--experiment-json",
        type=Path,
        default=None,
        help=(
            "JSON de un experimento previo (train): lee ``seed`` y ``validation_split`` "
            "para reproducir el mismo val que en el entrenamiento."
        ),
    )
    p.add_argument("--validation-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--ece-bins",
        type=int,
        default=15,
        metavar="M",
        help="Número de bins para el ECE (esperanza de error de calibración).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Ruta del informe JSON (por defecto artifacts/runs/calibration_<UTC>.json).",
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Ruta del informe Markdown (por defecto junto al JSON).",
    )
    p.add_argument(
        "--mlflow",
        action="store_true",
        help=(
            "Registrar calibración en MLflow (local ``file:<ml>/mlruns``; experimento "
            "``anemia-baseline-calibration`` o MLFLOW_EXPERIMENT_NAME_CALIBRATION)."
        ),
    )
    return p.parse_args()


def _read_split_params_from_experiment(path: Path) -> tuple[float, int, dict]:
    exp = json.loads(path.read_text(encoding="utf-8"))
    tc = exp.get("training_configuration") or {}
    vs = float(tc.get("validation_split", 0.2))
    seed = tc.get("seed")
    if seed is None:
        seed = exp.get("random_seed")
    if seed is None:
        raise SystemExit(
            f"El JSON {path} no contiene training_configuration.seed ni random_seed.",
        )
    return vs, int(seed), exp


def _f(v: object, *, nd: int = 6) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.{nd}f}"


def _markdown_report(payload: dict) -> str:
    cal = payload["calibration"]
    unc = payload["test_uncalibrated"]
    cali = payload["test_calibrated"]
    cmp_ = payload["comparison"]
    thr_u = (unc.get("thresholds_used") or {}).get("operational_threshold")
    thr_c = (cali.get("thresholds_used") or {}).get("operational_threshold")
    op_u = unc.get("at_operational_threshold") or {}
    op_c = cali.get("at_operational_threshold") or {}

    src = payload.get("source_experiment") or {}
    lines = [
        "# Informe: calibración por *temperature scaling* (post-hoc)",
        "",
        f"**Run ID:** `{payload['run_id']}`  ",
        f"**Marca de tiempo (UTC):** `{payload['timestamp_utc']}`",
        "",
        "## 1. Alcance y supuestos",
        "",
        "Este paso **no reentrena** la red convolucional ni altera sus pesos: se reutiliza el "
        "modelo guardado en disco y solo se aprende un parámetro escalar **T** que repesca las "
        "probabilidades de salida (misma decisión de ranking que una sigmoide con temperatura "
        "sobre el logit implícito).",
        "",
        "**Motivación (cribado clínico):** las probabilidades *raw* suelen estar mal calibradas "
        "(frecuencias empíricas ≠ probabilidades predichas). Ajustar **T** en validación mejora "
        "la **interpretabilidad** del score como grado de confianza, sin cambiar la capacidad "
        "discriminativa global (AUC-ROC es invariante ante transformaciones monótonas del score).",
        "",
        "## 2. Modelo y trazabilidad",
        "",
        f"- **Archivo .keras:** `{payload['model_path']}`",
    ]
    if src.get("experiment_json_path"):
        lines.extend(
            [
                f"- **JSON de experimento de referencia:** `{src.get('experiment_json_path')}`",
                f"- **run_id entrenamiento:** `{src.get('source_run_id', 'N/A')}`",
            ],
        )
    dlt = cmp_.get("delta_calibrated_minus_uncalibrated") or {}
    lines.extend(
        [
            "",
            "## 3. Ajuste de temperatura (validación)",
            "",
            f"- **Conjunto:** subconjunto **val** estratificado desde `--train-dir` "
            f"(``validation_split={cal['validation_split']}``, ``seed={cal['seed_used_for_val_split']}``), "
            "misma lógica que en ``train.py`` / ``load_train_val_datasets``.",
            f"- **Muestras en validación:** {cal['n_validation_samples']}",
            f"- **Temperatura aprendida ``T``:** **{cal['temperature_T']}**",
            f"- **NLL media (BCE) en val — antes (``T=1``):** {_f(cal.get('mean_nll_validation_before_T'))}",
            f"- **NLL media (BCE) en val — después (``T`` óptimo):** {_f(cal.get('mean_nll_validation_after_T'))}",
            f"- **Bins ECE (informe test):** {cal.get('ece_bins', 15)}",
            "",
            "## 4. Evaluación en test: sin calibrar vs calibrado",
            "",
            "| Métrica | Sin calibrar | Calibrado (``T``) |",
            "|---------|----------------|-------------------|",
            f"| AUC (ROC) | {_f(unc.get('auc'))} | {_f(cali.get('auc'))} |",
            f"| Loss (BCE media) | {_f(unc.get('loss'))} | {_f(cali.get('loss'))} |",
            f"| Brier | {_f(unc.get('brier_score'))} | {_f(cali.get('brier_score'))} |",
            f"| ECE ({unc.get('ece_bins', 15)} bins) | {_f(unc.get('expected_calibration_error'))} | "
            f"{_f(cali.get('expected_calibration_error'))} |",
            "",
            f"- **Δ (cal − sin calibrar):** BCE media **{_f(dlt.get('loss_bce_mean'))}**, "
            f"Brier **{_f(dlt.get('brier_score'))}**, ECE **{_f(dlt.get('expected_calibration_error'))}** "
            "(Brier y ECE suelen **bajar** al mejorar la calibración).",
            "",
            "### Umbral operacional (Youden en test)",
            "",
            "| | Sin calibrar | Calibrado |",
            "|--|--------------|-----------|",
            f"| **τ (Youden)** | **{_f(thr_u)}** | **{_f(thr_c)}** |",
            f"| Precision @τ | {_f(op_u.get('precision'))} | {_f(op_c.get('precision'))} |",
            f"| Recall @τ | {_f(op_u.get('recall'))} | {_f(op_c.get('recall'))} |",
            f"| Accuracy @τ | {_f(op_u.get('accuracy'))} | {_f(op_c.get('accuracy'))} |",
            f"| Youden J @τ | {_f(op_u.get('youden_j'))} | {_f(op_c.get('youden_j'))} |",
            "",
            f"_Nota: {cmp_.get('ranking_note', '')}_",
            "",
            "## 5. Referencia teórica τ = 0.5 (solo bibliográfica)",
            "",
            "| | Sin calibrar | Calibrado |",
            "|--|--------------|-----------|",
        ],
    )
    m5u = unc.get("at_threshold_0_5") or {}
    m5c = cali.get("at_threshold_0_5") or {}
    lines.extend(
        [
            f"| Precision | {_f(m5u.get('precision'))} | {_f(m5c.get('precision'))} |",
            f"| Recall | {_f(m5u.get('recall'))} | {_f(m5c.get('recall'))} |",
            f"| Accuracy | {_f(m5u.get('accuracy'))} | {_f(m5c.get('accuracy'))} |",
            "",
        ],
    )
    lines.append("## 6. Resumen ejecutivo")
    lines.append("")
    lines.append(
        f"- **T** = **{cal['temperature_T']}** (validación). "
        "Úsese junto con las métricas de fiabilidad (Brier, ECE) para discutir el score en tesis."
    )
    lines.append(
        f"- **Pérdida BCE en test** pasa de **{_f(unc.get('loss'))}** a **{_f(cali.get('loss'))}** "
        "tras calibrar (objetivo típico de *temperature scaling*)."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    run_id = f"calibration_{started.strftime('%Y%m%dT%H%M%SZ')}"
    ts = started.isoformat()

    model_path = args.model_path.expanduser().resolve()
    train_dir = args.train_dir.expanduser().resolve()
    test_dir = args.test_dir.expanduser().resolve()

    source_exp: dict | None = None
    exp_path_str: str | None = None
    validation_split = float(args.validation_split)
    seed = int(args.seed)

    if args.experiment_json:
        exp_path = args.experiment_json.expanduser().resolve()
        if not exp_path.is_file():
            raise SystemExit(f"No existe --experiment-json: {exp_path}")
        validation_split, seed, source_exp = _read_split_params_from_experiment(exp_path)
        exp_path_str = str(exp_path)

    if not model_path.is_file():
        raise SystemExit(f"No existe --model-path: {model_path}")
    if not train_dir.is_dir():
        raise SystemExit(f"No existe --train-dir: {train_dir}")
    if not test_dir.is_dir():
        raise SystemExit(f"No existe --test-dir: {test_dir}")

    model = keras.models.load_model(model_path)

    val_ds, val_counts, _class_names = load_validation_dataset(
        train_dir,
        validation_split=validation_split,
        seed=seed,
    )
    y_val, p_val = collect_binary_predictions(model, val_ds)
    n_val = int(y_val.size)
    pos_v = int((y_val == 1).sum())
    neg_v = n_val - pos_v

    T, fit_diag = fit_temperature_scaling_on_probabilities(y_val, p_val)
    if pos_v == 0 or neg_v == 0:
        fit_diag["warning_validation_single_class"] = (
            "El subconjunto de validación no tiene ambas clases; el ajuste de T puede ser poco informativo."
        )

    test_ds_eval = load_test_dataset(test_dir)
    raw_eval = model.evaluate(test_ds_eval, return_dict=True, verbose=1)
    loss_unc = float(raw_eval["loss"])
    auc_unc_keras = float(raw_eval["auc"])

    test_ds_pred = load_test_dataset(test_dir)
    y_test, p_test = collect_binary_predictions(model, test_ds_pred)
    p_cal = apply_temperature_scaling(p_test, T)

    loss_cal = mean_binary_cross_entropy(y_test, p_cal)
    auc_cal_keras = auc_roc_keras(y_test, p_cal)

    unc_base = build_threshold_evaluation_results(
        loss=loss_unc,
        auc_val=auc_unc_keras,
        y_true=y_test,
        y_prob=p_test,
    )
    cal_base = build_threshold_evaluation_results(
        loss=loss_cal,
        auc_val=auc_cal_keras,
        y_true=y_test,
        y_prob=p_cal,
    )
    unc = enrich_binary_eval_with_calibration_metrics(
        unc_base,
        y_test,
        p_test,
        n_ece_bins=args.ece_bins,
    )
    cali = enrich_binary_eval_with_calibration_metrics(
        cal_base,
        y_test,
        p_cal,
        n_ece_bins=args.ece_bins,
    )

    thr_u = float((unc_base.get("thresholds_used") or {}).get("operational_threshold", 0.5))
    thr_c = float((cal_base.get("thresholds_used") or {}).get("operational_threshold", 0.5))
    ranking_note = (
        "La curva ROC (y por tanto AUC y el Youden J máximo) es la misma para scores "
        "monótonamente transformados; el **valor de τ** en el eje de probabilidad cambia "
        "al aplicar ``T``, pero las decisiones en el óptimo de Youden coinciden con las del score raw."
    )

    source_block: dict[str, object] = {
        "experiment_json_path": exp_path_str,
        "source_run_id": (source_exp or {}).get("run_id"),
        "model_version_training": (source_exp or {}).get("model_version"),
    }

    payload: dict[str, object] = {
        "run_id": run_id,
        "timestamp_utc": ts,
        "software_versions": _software_versions(),
        "model_path": str(model_path),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "source_experiment": source_block,
        "calibration": {
            "method": "temperature_scaling",
            "thesis_summary": (
                "Calibración post-hoc: solo se estima un escalar T sobre los logits implícitos "
                "de las probabilidades ya predichas; el backbone MobileNetV2 y el cabezal "
                "permanecen idénticos al modelo entrenado."
            ),
            "temperature_T": float(T),
            "validation_split": float(validation_split),
            "seed_used_for_val_split": int(seed),
            "n_validation_samples": n_val,
            "validation_class_counts": {str(k): int(v) for k, v in val_counts.items()},
            "mean_nll_validation_before_T": float(fit_diag["mean_nll_validation_before_T"]),
            "mean_nll_validation_after_T": float(fit_diag["mean_nll_validation_after_T"]),
            "fit_diagnostics": {k: v for k, v in fit_diag.items() if k not in (
                "mean_nll_validation_before_T",
                "mean_nll_validation_after_T",
            )},
            "ece_bins": int(args.ece_bins),
        },
        "test_uncalibrated": unc,
        "test_calibrated": cali,
        "comparison": {
            "operational_threshold_youden_test": {
                "uncalibrated": thr_u,
                "calibrated": thr_c,
                "delta_calibrated_minus_uncalibrated": float(thr_c - thr_u),
            },
            "delta_calibrated_minus_uncalibrated": {
                "loss_bce_mean": float(cali["loss"] - unc["loss"]),
                "brier_score": float(cali["brier_score"] - unc["brier_score"]),
                "expected_calibration_error": float(
                    cali["expected_calibration_error"] - unc["expected_calibration_error"],
                ),
            },
            "ranking_note": ranking_note,
        },
    }

    out_json = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else (RUNS_DIR / f"{run_id}.json")
    )
    out_md = (
        args.output_md.expanduser().resolve()
        if args.output_md
        else out_json.with_suffix(".md")
    )

    write_json(out_json, payload)
    write_text(out_md, _markdown_report(payload))

    mlflow_ok = False
    if args.mlflow:
        from baseline.mlflow_logging import safe_log_calibration_run

        mlflow_ok = safe_log_calibration_run(
            payload,
            ml_root=_ML_ROOT,
            report_json=out_json,
            report_md=out_md,
        )

    print(f"\nTemperatura T (validación): {T}")
    print(f"JSON: {out_json}")
    print(f"Markdown: {out_md}")
    if args.mlflow and mlflow_ok:
        print("MLflow: run de calibración registrado (ver ml/mlruns o MLFLOW_TRACKING_URI).")


if __name__ == "__main__":
    main()
