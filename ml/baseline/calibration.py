"""Calibración post-hoc: temperature scaling, Platt (producción) e isotonic PAVA (ablación).

No modifica pesos del modelo. Temperature: ``p' = sigmoid(logit(p) / T)``.
Platt: ``p' = sigmoid(a * logit(p) + b)`` (sin sklearn).
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


def apply_platt_scaling(y_prob: np.ndarray, platt_a: float, platt_b: float) -> np.ndarray:
    """``p' = sigmoid(a * logit(p) + b)``. ``(a=1, b=0)`` es la identidad."""
    z = float(platt_a) * _safe_logit(np.asarray(y_prob, dtype=np.float64)) + float(platt_b)
    z = np.clip(z, -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float64, copy=False)


def _pava_increasing(values: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators: proyección no decreciente (media por bloque)."""
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(y.size)
    if n == 0:
        return y
    means: list[float] = []
    counts: list[int] = []
    for val in y:
        means.append(float(val))
        counts.append(1)
        while len(means) >= 2 and means[-2] > means[-1] + 1e-15:
            c0, c1 = counts[-2], counts[-1]
            merged = (means[-2] * c0 + means[-1] * c1) / (c0 + c1)
            means.pop()
            counts.pop()
            means[-1] = merged
            counts[-1] = c0 + c1
    out = np.empty(n, dtype=np.float64)
    idx = 0
    for mean, count in zip(means, counts, strict=True):
        out[idx : idx + count] = mean
        idx += count
    return out


def fit_isotonic_regression_on_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Isotonic (PAVA) en validación. Devuelve nudos ``(x, y)`` para ``np.interp``.

    Ablación: no es el default móvil (tabla vs 2 floats de Platt).
    """
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(y_prob, dtype=np.float64).reshape(-1), 1e-7, 1.0 - 1e-7)
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    fitted = _pava_increasing(y[order])
    x_knots, first_idx = np.unique(p_sorted, return_index=True)
    y_knots = fitted[first_idx]
    return x_knots, y_knots, {
        "n_knots": int(x_knots.size),
        "y_min": float(np.min(y_knots)) if y_knots.size else 0.0,
        "y_max": float(np.max(y_knots)) if y_knots.size else 0.0,
    }


def apply_isotonic_interpolation(
    y_prob: np.ndarray,
    x_knots: np.ndarray,
    y_knots: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x_knots, dtype=np.float64).reshape(-1)
    y = np.asarray(y_knots, dtype=np.float64).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return p.astype(np.float64, copy=False)
    if x.size == 1:
        return np.full(p.shape, float(y[0]), dtype=np.float64)
    return np.interp(p, x, y).astype(np.float64, copy=False)


def fit_platt_scaling_on_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    a_min: float = 0.05,
    a_max: float = 20.0,
    b_min: float = -8.0,
    b_max: float = 8.0,
    a_points: int = 48,
    b_points: int = 41,
    refine_a: int = 24,
    refine_b: int = 21,
) -> tuple[float, float, dict[str, Any]]:
    """Busca ``(a, b)`` que minimizan NLL en validación (rejilla 2D, sin sklearn)."""
    y = np.asarray(y_true, dtype=np.int32).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    nll_before = mean_binary_cross_entropy(y, p)

    a_grid = np.concatenate(
        (-np.geomspace(float(a_min), float(a_max), int(a_points) // 2),
         np.geomspace(float(a_min), float(a_max), int(a_points) - int(a_points) // 2))
    )
    b_grid = np.linspace(float(b_min), float(b_max), int(b_points))
    best_a, best_b, best_nll = 1.0, 0.0, float("inf")
    for a in a_grid:
        for b in b_grid:
            q = apply_platt_scaling(p, float(a), float(b))
            nll = mean_binary_cross_entropy(y, q)
            if nll < best_nll - 1e-12:
                best_nll = nll
                best_a = float(a)
                best_b = float(b)

    a_lo = best_a / 1.5 if best_a > 0 else best_a * 1.5
    a_hi = best_a * 1.5 if best_a > 0 else best_a / 1.5
    if a_lo > a_hi:
        a_lo, a_hi = a_hi, a_lo
    if abs(best_a) < 1e-8:
        a_lo, a_hi = -1.0, 1.0
    b_span = max(0.75, 0.25 * abs(best_b) if abs(best_b) > 1e-8 else 1.0)
    fine_a = np.linspace(a_lo, a_hi, int(refine_a))
    fine_b = np.linspace(best_b - b_span, best_b + b_span, int(refine_b))
    for a in fine_a:
        for b in fine_b:
            q = apply_platt_scaling(p, float(a), float(b))
            nll = mean_binary_cross_entropy(y, q)
            if nll < best_nll - 1e-12:
                best_nll = nll
                best_a = float(a)
                best_b = float(b)

    q_final = apply_platt_scaling(p, best_a, best_b)
    return best_a, best_b, {
        "mean_nll_validation_before_platt": float(nll_before),
        "mean_nll_validation_after_platt": float(mean_binary_cross_entropy(y, q_final)),
        "a_grid_abs_min": float(a_min),
        "a_grid_abs_max": float(a_max),
        "b_grid_min": float(b_min),
        "b_grid_max": float(b_max),
    }


def apply_named_calibration(
    y_prob: np.ndarray,
    method: str,
    *,
    temperature: float = 1.0,
    platt_a: float = 1.0,
    platt_b: float = 0.0,
    isotonic_x: np.ndarray | None = None,
    isotonic_y: np.ndarray | None = None,
) -> np.ndarray:
    resolved = str(method).strip().lower()
    if resolved == "platt":
        return apply_platt_scaling(y_prob, platt_a, platt_b)
    if resolved == "isotonic":
        if isotonic_x is None or isotonic_y is None:
            raise ValueError("isotonic requiere isotonic_x e isotonic_y")
        return apply_isotonic_interpolation(y_prob, isotonic_x, isotonic_y)
    return apply_temperature_scaling(y_prob, temperature)


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
