"""Calibración post-hoc por *temperature scaling* (Guo et al., 2017).

No modifica pesos del modelo: solo ajusta un escalar ``T > 0`` sobre los logits
implícitos de las probabilidades ya predichas: ``p' = sigmoid(logit(p) / T)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf


def _safe_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def apply_temperature_scaling(y_prob: np.ndarray, temperature: float) -> np.ndarray:
    """Probabilidades calibradas con temperatura ``T`` (``T=1`` deja ``p`` casi sin cambio)."""
    T = float(max(temperature, 1e-8))
    z = _safe_logit(np.asarray(y_prob, dtype=np.float64))
    return (1.0 / (1.0 + np.exp(-z / T))).astype(np.float64, copy=False)


def mean_binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(y_prob, dtype=np.float64).reshape(-1), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier_score_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """
    ECE (esperanza de |exactitud − confianza| por bin de probabilidad predicha).

    ``confianza`` = media de ``p`` en el bin; ``exactitud`` = proporción de positivos
    en el bin (convención estándar para diagramas de fiabilidad binarios).
    """
    y = np.asarray(y_true, dtype=np.int32).reshape(-1)
    p = np.clip(np.asarray(y_prob, dtype=np.float64).reshape(-1), 0.0, 1.0)
    n = int(y.size)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for b in range(int(n_bins)):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if b == int(n_bins) - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask].astype(np.float64)))
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def auc_roc_keras(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC-ROC alineada con la métrica ``AUC`` de Keras/TensorFlow (curva ROC)."""
    m = tf.keras.metrics.AUC(curve="ROC")
    m.reset_state()
    m.update_state(
        np.asarray(y_true, dtype=np.float32).reshape(-1),
        np.asarray(y_prob, dtype=np.float32).reshape(-1),
    )
    return float(m.result().numpy())


def fit_temperature_scaling_on_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    t_min: float = 5e-2,
    t_max: float = 50.0,
    grid_points: int = 160,
    refine_points: int = 96,
) -> tuple[float, dict[str, Any]]:
    """
    Busca ``T`` que minimiza la NLL (BCE media) en validación.

    Búsqueda en rejilla log-uniforme + refinamiento local (sin dependencias extra).
    """
    y = np.asarray(y_true, dtype=np.int32).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    nll_before = mean_binary_cross_entropy(y, p)

    grid = np.geomspace(float(t_min), float(t_max), int(grid_points))
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        q = apply_temperature_scaling(p, float(T))
        nll = mean_binary_cross_entropy(y, q)
        if nll < best_nll - 1e-12:
            best_nll = nll
            best_T = float(T)

    lo = max(float(t_min), best_T / 1.35)
    hi = min(float(t_max), best_T * 1.35)
    fine = np.geomspace(lo, hi, int(refine_points))
    for T in fine:
        q = apply_temperature_scaling(p, float(T))
        nll = mean_binary_cross_entropy(y, q)
        if nll < best_nll - 1e-12:
            best_nll = nll
            best_T = float(T)

    q_final = apply_temperature_scaling(p, best_T)
    nll_after = mean_binary_cross_entropy(y, q_final)

    return best_T, {
        "mean_nll_validation_before_T": float(nll_before),
        "mean_nll_validation_after_T": float(nll_after),
        "temperature_grid_t_min": float(t_min),
        "temperature_grid_t_max": float(t_max),
        "grid_points": int(grid_points),
        "refine_points": int(refine_points),
    }


def enrich_binary_eval_with_calibration_metrics(
    base: dict[str, Any],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_ece_bins: int = 15,
) -> dict[str, Any]:
    """Copia el dict de evaluación y añade Brier y ECE (no altera umbrales ni AUC)."""
    out = dict(base)
    out["brier_score"] = brier_score_binary(y_true, y_prob)
    out["expected_calibration_error"] = expected_calibration_error_binary(
        y_true,
        y_prob,
        n_bins=int(n_ece_bins),
    )
    out["ece_bins"] = int(n_ece_bins)
    return out
