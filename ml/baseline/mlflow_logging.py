"""Integración opcional con MLflow.

Por defecto la URI es ``file:<directorio_ml>/mlruns``. Se puede sobrescribir con la variable de
entorno ``MLFLOW_TRACKING_URI``. Experimentos por defecto: ``anemia-baseline-train``,
``anemia-baseline-calibration``, configurables mediante ``MLFLOW_EXPERIMENT_NAME`` y
``MLFLOW_EXPERIMENT_NAME_CALIBRATION``.

Los puntos de entrada ``safe_log_*`` **no propagan excepciones**: un fallo en MLflow no interrumpe
entrenamiento ni calibración (solo escribe en stderr).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _log_existing_files_as_artifacts(mlflow: Any, paths: list[Path]) -> None:
    """Adjunta ficheros al run activo (copias en el artefact store)."""
    for p in paths:
        rp = p.resolve()
        if rp.is_file():
            mlflow.log_artifact(str(rp))


def default_tracking_uri(ml_root: Path) -> str:
    """Store local bajo ``<ml_root>/mlruns``."""
    root = ml_root.resolve()
    mlruns = root / "mlruns"
    mlruns.mkdir(parents=True, exist_ok=True)
    return f"file:{mlruns}"


def _resolved_tracking_uri(ml_root: Path) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or default_tracking_uri(ml_root)


def log_train_experiment(
    experiment: dict[str, Any],
    *,
    ml_root: Path,
    report_json: Path,
    report_md: Path,
    model_path: Path,
) -> None:
    import mlflow

    mlflow.set_tracking_uri(_resolved_tracking_uri(ml_root))
    exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "anemia-baseline-train")
    mlflow.set_experiment(exp_name)

    rid = str(experiment.get("run_id", "unknown"))
    tc = experiment.get("training_configuration") or {}
    cd = experiment.get("class_distribution") or {}
    ci = experiment.get("class_imbalance") or {}
    osb = experiment.get("oversampling") or {}
    res = experiment.get("results") or {}

    with mlflow.start_run(run_name=f"train_{rid}"):
        mlflow.set_tag("pipeline", "train")
        mlflow.set_tag("run_id", rid)
        mlflow.set_tag("model_version", str(experiment.get("model_version", "")))

        mlflow.log_param("random_seed", int(experiment.get("random_seed", 0)))
        mlflow.log_param("validation_split", float(tc.get("validation_split", 0.0)))
        mlflow.log_param("head_epochs", int(tc.get("head_epochs", 0)))
        mlflow.log_param("fine_tune_epochs", int(tc.get("fine_tune_epochs", 0)))
        mlflow.log_param("head_learning_rate", float(tc.get("head_learning_rate", 0.0)))
        mlflow.log_param("fine_tune_learning_rate", float(tc.get("fine_tune_learning_rate", 0.0)))
        mlflow.log_param(
            "fine_tune_freeze_up_to_layer",
            int(tc.get("fine_tune_freeze_up_to_layer", 0)),
        )
        mlflow.log_param("demo", bool(tc.get("demo", False)))
        aug = str(tc.get("augmentation_summary", "") or "")
        mlflow.log_param("augmentation_summary", aug[:500])

        mlflow.log_param("oversample_positive_cli", bool(tc.get("oversample_positive")))
        mlflow.log_param("no_class_weight_cli", bool(tc.get("no_class_weight")))
        mlflow.log_param("class_weight_enabled_in_fit", bool(ci.get("class_weight_enabled_in_fit")))
        mlflow.log_param("oversampling_used", bool(ci.get("oversampling_used")))
        mlflow.log_param("oversampling_requested", bool(ci.get("oversampling_requested")))
        mlflow.log_param("oversampling_applied_effective", bool(osb.get("applied")))

        _bft = tc.get("backbone_fine_tuning")
        bft = _bft if isinstance(_bft, dict) else {}
        mlflow.log_param("fine_tuning_enabled", bool(bft.get("enabled")))

        mlflow.log_param("dist.class_name_0", str(cd.get("class_name_0", "")))
        mlflow.log_param("dist.class_name_1", str(cd.get("class_name_1", "")))
        mlflow.log_param("dist.negative_total", int(cd.get("negative_total", 0)))
        mlflow.log_param("dist.positive_total", int(cd.get("positive_total", 0)))

        sct = cd.get("split_class_counts") or {}
        mlflow.log_param("dist.split_class_counts_json", json.dumps(sct, sort_keys=True))

        mlflow.log_param(
            "dist.train_subset_counts_json",
            json.dumps(ci.get("train_subset_counts") or {}, sort_keys=True),
        )
        cw_js = json.dumps(ci.get("computed_class_weights") or {}, sort_keys=True)
        mlflow.log_param("class.computed_weights_json", cw_js[:500])

        bl = str(tc.get("baseline_experiment_json") or "")
        mlflow.log_param("baseline_experiment_json_path", bl)

        mlflow.log_param("artifact.model_path", str(model_path.resolve()))
        mlflow.log_param("artifact.experiment_json", str(report_json.resolve()))
        mlflow.log_param("artifact.experiment_md", str(report_md.resolve()))

        if res.get("loss") is not None and res.get("auc") is not None:
            mlflow.log_metric("test_loss", float(res["loss"]))
            mlflow.log_metric("test_auc", float(res["auc"]))
            mop = res.get("at_operational_threshold") or {}
            if mop:
                mlflow.log_metric("test_precision_operational", float(mop.get("precision", 0.0)))
                mlflow.log_metric("test_recall_operational", float(mop.get("recall", 0.0)))
                mlflow.log_metric("test_accuracy_operational", float(mop.get("accuracy", 0.0)))
                mlflow.log_metric("threshold_operational_youden", float(mop.get("threshold", 0.0)))
                mlflow.log_metric("youden_j_operational", float(mop.get("youden_j", 0.0)))
            m05 = res.get("at_threshold_0_5") or {}
            if m05:
                mlflow.log_metric("test_precision_ref_0_5", float(m05.get("precision", 0.0)))
                mlflow.log_metric("test_recall_ref_0_5", float(m05.get("recall", 0.0)))
                mlflow.log_metric("test_accuracy_ref_0_5", float(m05.get("accuracy", 0.0)))

        artifact_paths: list[Path] = [report_json, report_md]
        if model_path.is_file():
            artifact_paths.append(model_path)
        _log_existing_files_as_artifacts(mlflow, artifact_paths)


def log_calibration_run(
    payload: dict[str, Any],
    *,
    ml_root: Path,
    report_json: Path,
    report_md: Path,
) -> None:
    import mlflow

    mlflow.set_tracking_uri(_resolved_tracking_uri(ml_root))
    exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME_CALIBRATION", "anemia-baseline-calibration")
    mlflow.set_experiment(exp_name)

    rid = str(payload.get("run_id", "unknown"))
    cal = payload.get("calibration") or {}
    unc = payload.get("test_uncalibrated") or {}
    cali = payload.get("test_calibrated") or {}
    cmp_ = payload.get("comparison") or {}
    thr_blk = cmp_.get("operational_threshold_youden_test") or {}

    with mlflow.start_run(run_name=f"calibration_{rid}"):
        mlflow.set_tag("pipeline", "calibration")
        mlflow.set_tag("run_id", rid)

        mlflow.log_param("temperature", float(cal.get("temperature_T", 0.0)))
        mlflow.log_param("validation_split", float(cal.get("validation_split", 0.0)))
        mlflow.log_param("seed_used_for_val_split", int(cal.get("seed_used_for_val_split", 0)))
        mlflow.log_param("ece_bins", int(cal.get("ece_bins", 0)))

        mlflow.log_metric("brier_uncalibrated", float(unc.get("brier_score", 0.0)))
        mlflow.log_metric("brier_calibrated", float(cali.get("brier_score", 0.0)))
        mlflow.log_metric("ece_uncalibrated", float(unc.get("expected_calibration_error", 0.0)))
        mlflow.log_metric("ece_calibrated", float(cali.get("expected_calibration_error", 0.0)))

        u0 = float(thr_blk.get("uncalibrated", 0.0))
        u1 = float(thr_blk.get("calibrated", 0.0))
        mlflow.log_metric("threshold_operational_uncalibrated", u0)
        mlflow.log_metric("threshold_operational_calibrated", u1)

        mlflow.log_param("artifact.calibration_json", str(report_json.resolve()))
        mlflow.log_param("artifact.calibration_md", str(report_md.resolve()))
        ml_path_raw = str(payload.get("model_path", "") or "")
        mlflow.log_param("model_path", ml_path_raw)

        artifact_paths: list[Path] = [report_json, report_md]
        mp = Path(ml_path_raw).expanduser().resolve()
        if mp.is_file():
            artifact_paths.append(mp)
        _log_existing_files_as_artifacts(mlflow, artifact_paths)


def safe_log_train_experiment(
    experiment: dict[str, Any],
    *,
    ml_root: Path,
    report_json: Path,
    report_md: Path,
    model_path: Path,
) -> bool:
    """
    Ejecuta ``log_train_experiment`` sin propagar errores.

    Returns:
        True si el run se registró sin error.
    """
    try:
        log_train_experiment(
            experiment,
            ml_root=ml_root,
            report_json=report_json,
            report_md=report_md,
            model_path=model_path,
        )
        return True
    except ImportError:
        print(
            "[mlflow] Paquete no instalado; omitiendo tracking. "
            "Instale con: cd ml && pip install -r requirements.txt",
            file=sys.stderr,
        )
        return False
    except Exception as e:
        print(f"[mlflow] No se pudo registrar el experimento (se ignora): {e}", file=sys.stderr)
        return False


def safe_log_calibration_run(
    payload: dict[str, Any],
    *,
    ml_root: Path,
    report_json: Path,
    report_md: Path,
) -> bool:
    """Ejecuta ``log_calibration_run`` sin propagar errores."""
    try:
        log_calibration_run(
            payload,
            ml_root=ml_root,
            report_json=report_json,
            report_md=report_md,
        )
        return True
    except ImportError:
        print(
            "[mlflow] Paquete no instalado; omitiendo tracking. "
            "Instale con: cd ml && pip install -r requirements.txt",
            file=sys.stderr,
        )
        return False
    except Exception as e:
        print(f"[mlflow] No se pudo registrar la calibración (se ignora): {e}", file=sys.stderr)
        return False
