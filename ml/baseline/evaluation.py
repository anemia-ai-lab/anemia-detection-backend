"""Métricas de evaluación binaria por umbral e índice de Youden (ROC)."""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras


def collect_binary_predictions(
    model: keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Acumula ``y_true`` (0/1) y probabilidades ``y_prob`` en el orden del ``dataset``."""
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for xb, yb in dataset:
        pb = model(xb, training=False).numpy()
        ys.append(np.asarray(yb).reshape(-1))
        ps.append(np.asarray(pb).reshape(-1))
    y_true = np.concatenate(ys).astype(np.int32, copy=False)
    y_prob = np.concatenate(ps).astype(np.float64, copy=False)
    return y_true, y_prob


def binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Accuracy, precisión y recall (sensibilidad) con regla ``y_prob >= threshold``."""
    y_true = np.asarray(y_true).astype(int, copy=False)
    y_prob = np.asarray(y_prob).astype(np.float64, copy=False)
    y_pred = (y_prob >= threshold).astype(int, copy=False)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
    }


def youden_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Umbrales candidatos en los puntos del ROC (probabilidades predichas y 0/1).
    Maximiza el índice de Youden ``J = TPR - FPR`` (equiv. a sensibilidad + especificidad - 1).
    """
    y_true = np.asarray(y_true).astype(int, copy=False)
    y_prob = np.asarray(y_prob).astype(np.float64, copy=False)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return 0.5, 0.0

    candidates = np.unique(np.concatenate(([0.0], np.sort(y_prob), [1.0])))
    best_t = 0.5
    best_j = -1.0
    for t in candidates:
        y_pred = (y_prob >= t).astype(int, copy=False)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tpr = tp / pos if pos else 0.0
        fpr = fp / neg if neg else 0.0
        j = tpr - fpr
        if j > best_j + 1e-15 or (abs(j - best_j) <= 1e-15 and t < best_t):
            best_j = j
            best_t = float(t)
    return best_t, float(best_j)


CLINICAL_INTERPRETATION: dict[str, str] = {
    "why_threshold_0_5_not_default": (
        "Con datos clínicos **desbalanceados** (pocos positivos frente a negativos), fijar 0.5 "
        "sobre la salida sigmoide suele **no** coincidir con un punto razonable de la curva ROC: "
        "puede inflar la exactitud global mientras **deja sin detectar** demasiados positivos "
        "reales. El coste de un falso negativo (pasar anemia por alto) y el de un falso positivo "
        "no son simétricos, por lo que el corte 0.5 no debe tomarse como decisión operativa por defecto."
    ),
    "why_prioritize_recall": (
        "En un contexto de **cribado** orientado a detectar anemia, se prioriza la "
        "**sensibilidad (recall)** sobre la mera precisión global: captar la mayoría de casos "
        "positivos reduce riesgo clínico; los falsos positivos suelen poder aclararse con "
        "hemograma u otras pruebas. El índice de Youden en ROC equilibra TPR y FPR y ofrece un "
        "umbral de trabajo más informativo que 0.5 bajo desbalance."
    ),
    "operational_threshold_use": (
        "El **umbral operacional** (por defecto el de máximo índice de Youden en el conjunto de "
        "evaluación) es el que se documenta como referencia para **interpretación clínica y "
        "predicciones** en este informe; el umbral 0.5 se mantiene solo como **referencia teórica** "
        "habitual en modelos binarios."
    ),
}


def build_threshold_evaluation_results(
    *,
    loss: float,
    auc_val: float,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """
    Construye el bloque ``results`` de evaluación en test: AUC, umbral teórico 0.5 (referencia),
    umbral operacional ROC-Youden (métricas principales) y texto de interpretación clínica.
    """
    at_05 = binary_metrics_at_threshold(y_true, y_prob, 0.5)
    t_star, j_max = youden_optimal_threshold(y_true, y_prob)
    at_op = binary_metrics_at_threshold(y_true, y_prob, t_star)
    return {
        "loss": float(loss),
        "auc": float(auc_val),
        "auc_note": "AUC de Keras (ROC); independiente del umbral de decisión.",
        "clinical_interpretation": dict(CLINICAL_INTERPRETATION),
        "thresholds_used": {
            "theoretical_default_threshold": 0.5,
            "theoretical_role": (
                "Referencia estándar (probabilidad ≥ 0.5); solo para comparación bibliográfica, "
                "no como umbral operacional en este estudio."
            ),
            "operational_threshold": float(t_star),
            "operational_threshold_source": "roc_youden_max_j",
            "youden_j_at_operational_threshold": float(j_max),
            "operational_for_predictions_note": CLINICAL_INTERPRETATION["operational_threshold_use"],
            "fixed_decision_threshold": 0.5,
            "roc_youden_optimal_threshold": float(t_star),
            "youden_j_max": float(j_max),
        },
        "at_operational_threshold": {
            "threshold": float(t_star),
            "precision": at_op["precision"],
            "recall": at_op["recall"],
            "accuracy": at_op["accuracy"],
            "youden_j": float(j_max),
            "source": "roc_youden_max_j",
            "is_primary_reporting_metrics": True,
        },
        "at_threshold_0_5": {
            **at_05,
            "role": "reference_only_theoretical_sigmoid_midpoint",
        },
        "at_youden_optimal": {
            "optimal_threshold": float(t_star),
            "youden_j": float(j_max),
            "accuracy": at_op["accuracy"],
            "precision": at_op["precision"],
            "recall": at_op["recall"],
        },
    }
