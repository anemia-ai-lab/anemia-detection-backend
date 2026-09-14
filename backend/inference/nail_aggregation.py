"""Agregación de probabilidades calibradas por uña (una mano).

Modos
-----
``max``
    Uña con mayor ``p_cal`` (rollback; cribado sensible, sesgo hacia positivo).
``median`` (default de producto)
    Mediana **conservadora en n par**: n impar → valor central; n par → el menor
    de los dos centrales (n=2 → mínimo). **No** se usa ``statistics.median()``
    de Python, que con dos valores promedia ([0.9, 0.2] → 0.55) y dejaría un
    falso positivo por encima del umbral operacional (~0.38).
``majority_2_of_3``
    Segundo más alto de ``p_cal`` (voto ≥2 sobre cualquier τ; n=2 → mínimo).

Con n ∈ {1, 2, 3} (los únicos conteos de este producto) ``median`` lower-even y
``majority_2_of_3`` son el mismo operador numérico. Se mantienen ambos nombres:
``median`` es el default; ``majority_2_of_3`` es el alias explícito de voto.

Empates: orden estable de dedos index < middle < ring < hand.
"""

from __future__ import annotations

from typing import Literal, Sequence

NailAggregationMode = Literal["max", "median", "majority_2_of_3"]

_FINGER_RANK: dict[str, int] = {
    "index": 0,
    "middle": 1,
    "ring": 2,
    "hand": 3,
}

_VALID_MODES: frozenset[str] = frozenset(("max", "median", "majority_2_of_3"))


def normalize_nail_aggregation_mode(mode: str) -> NailAggregationMode:
    raw = str(mode).strip().lower()
    if raw not in _VALID_MODES:
        raise ValueError(
            f"Agregación de uñas inválida: {mode!r}. "
            "Use max, median o majority_2_of_3."
        )
    return raw  # type: ignore[return-value]


def _finger_rank(finger: str) -> int:
    return _FINGER_RANK.get(str(finger), len(_FINGER_RANK))


def _sorted_indices_by_calibrated(
    calibrated: Sequence[float],
    fingers: Sequence[str],
) -> list[int]:
    n = len(calibrated)
    return sorted(
        range(n),
        key=lambda i: (float(calibrated[i]), _finger_rank(fingers[i] if i < len(fingers) else "")),
    )


def select_aggregated_nail_index(
    calibrated: Sequence[float],
    *,
    fingers: Sequence[str],
    mode: NailAggregationMode | str,
) -> int:
    """Índice de la uña elegida según ``mode``. ``calibrated`` no vacío."""
    n = len(calibrated)
    if n == 0:
        raise ValueError("No hay uñas para agregar.")
    if n != len(fingers):
        raise ValueError("calibrated y fingers deben tener la misma longitud.")
    resolved = normalize_nail_aggregation_mode(str(mode))
    order = _sorted_indices_by_calibrated(calibrated, fingers)
    if resolved == "max":
        return order[-1]
    if n % 2 == 1:
        # Valor central (n=1 → 0; n=3 → 1). Coincide con 2.º más alto si n=3.
        pick = n // 2
    else:
        # Lower-even: menor de los dos centrales (n=2 → 0 = mínimo).
        # No usar statistics.median() (promedio).
        pick = n // 2 - 1
    return order[pick]
