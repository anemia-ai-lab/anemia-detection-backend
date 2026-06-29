#!/usr/bin/env python3
"""
Resumen de métricas Prometheus de latencia por fase de POST /predict.

Requiere:
  SMOKE_BASE_URL
  METRICS_BEARER_TOKEN
"""

from __future__ import annotations

import os
import re
import sys

import httpx

_BUCKET_LINE = re.compile(
    r'predict_phase_duration_seconds_bucket\{phase="([^"]+)",le="([^"]+)"\}\s+(\S+)'
)
_COUNT_LINE = re.compile(r'predict_phase_duration_seconds_count\{phase="([^"]+)"\}\s+(\S+)')


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"ERROR: falta {name}", file=sys.stderr)
        sys.exit(1)
    return value


def _approx_quantile(buckets: list[tuple[float, float]], q: float) -> float | None:
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None
    target = total * q
    for le, count in buckets:
        if count >= target:
            return le
    return buckets[-1][0]


def main() -> None:
    base = _env("SMOKE_BASE_URL").rstrip("/")
    token = _env("METRICS_BEARER_TOKEN")
    r = httpx.get(
        f"{base}/metrics",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    )
    if r.status_code != 200:
        print(f"ERROR: GET /metrics HTTP {r.status_code}", file=sys.stderr)
        sys.exit(1)

    by_phase: dict[str, list[tuple[float, float]]] = {}
    counts: dict[str, float] = {}
    for line in r.text.splitlines():
        m = _BUCKET_LINE.match(line)
        if m:
            phase, le_s, count_s = m.groups()
            le = float("inf") if le_s == "+Inf" else float(le_s)
            by_phase.setdefault(phase, []).append((le, float(count_s)))
            continue
        m2 = _COUNT_LINE.match(line)
        if m2:
            counts[m2.group(1)] = float(m2.group(2))

    if not by_phase:
        print("No hay muestras predict_phase_duration_seconds (¿sin tráfico /predict aún?)")
        sys.exit(0)

    print(f"Baseline métricas → {base}\n")
    for phase in sorted(by_phase):
        buckets = sorted(by_phase[phase], key=lambda x: x[0])
        total = counts.get(phase, buckets[-1][1] if buckets else 0.0)
        p50 = _approx_quantile(buckets, 0.5)
        p95 = _approx_quantile(buckets, 0.95)
        p50_s = f"{p50:.3f}s" if p50 is not None and p50 != float("inf") else "n/a"
        p95_s = f"{p95:.3f}s" if p95 is not None and p95 != float("inf") else "n/a"
        print(f"  phase={phase:16} count={int(total):5}  p50≈{p50_s}  p95≈{p95_s}")

    print("\nRegistrar fecha y valores antes/después de cambios de performance.")


if __name__ == "__main__":
    main()
