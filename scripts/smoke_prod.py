#!/usr/bin/env python3
"""
Smoke E2E contra el API desplegado (Render prod por defecto).

Requiere variables de entorno:
  SMOKE_BASE_URL          — base sin barra final (default: URL pública Render)
  SMOKE_EMAIL / SMOKE_PASSWORD — usuario Supabase dedicado a smoke
  METRICS_BEARER_TOKEN    — mismo token que en Render (GET /metrics)

Salida: 0 si todos los pasos pasan; 1 en caso contrario.
"""

from __future__ import annotations

import os
import sys
from io import BytesIO

import httpx

DEFAULT_BASE_URL = "https://anemia-detection-backend.onrender.com"
VALID_RISKS = frozenset({"low", "medium", "high"})
TIMEOUT_DEFAULT = httpx.Timeout(30.0, read=120.0)
TIMEOUT_PREDICT = httpx.Timeout(30.0, read=180.0)


def _env(name: str, *, required: bool = False, default: str = "") -> str:
    value = os.environ.get(name, default).strip()
    if required and not value:
        print(f"ERROR: falta variable de entorno {name}", file=sys.stderr)
        sys.exit(1)
    return value


def _base_url() -> str:
    raw = _env("SMOKE_BASE_URL", default=DEFAULT_BASE_URL)
    return raw.rstrip("/")


def skin_patch_png() -> bytes:
    """PNG 32×32 tono piel; pasa heurística de uña en backend/inference/nail_presence.py."""
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (32, 32), (220, 180, 140)).save(buf, format="PNG")
    return buf.getvalue()


def _fail(step: str, detail: str, *, status: int | None = None) -> None:
    prefix = f"FAIL [{step}]"
    if status is not None:
        prefix += f" HTTP {status}"
    print(f"{prefix}: {detail}", file=sys.stderr)
    sys.exit(1)


def _ok(step: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"OK   [{step}]{suffix}")


def check_health(client: httpx.Client, base: str) -> None:
    step = "health"
    r = client.get(f"{base}/health")
    if r.status_code != 200:
        _fail(step, r.text, status=r.status_code)
    data = r.json()
    if data.get("status") != "ok":
        _fail(step, f"status={data.get('status')!r}")
    if not data.get("model_loaded"):
        _fail(step, "model_loaded=false")
    if data.get("model_version") != "v2.0":
        _fail(step, f"model_version={data.get('model_version')!r}")
    _ok(step, "model_loaded=true model_version=v2.0")


def obtain_access_token(
    client: httpx.Client,
    base: str,
    email: str,
    password: str,
) -> str:
    step = "auth/login"
    r = client.post(f"{base}/auth/login", json={"email": email, "password": password})
    if r.status_code == 401:
        reg = client.post(
            f"{base}/auth/register",
            json={"email": email, "password": password},
        )
        if reg.status_code not in (200, 201):
            _fail(
                "auth/register",
                reg.text,
                status=reg.status_code,
            )
        r = client.post(f"{base}/auth/login", json={"email": email, "password": password})
    if r.status_code != 200:
        _fail(step, r.text, status=r.status_code)
    body = r.json()
    tokens = body.get("tokens") or {}
    token = tokens.get("access_token")
    if not token:
        _fail(step, "respuesta sin tokens.access_token")
    _ok(step)
    return str(token)


def check_profile(client: httpx.Client, base: str, token: str) -> None:
    step = "auth/me/profile"
    r = client.get(
        f"{base}/auth/me/profile",
        headers={"Authorization": f"Bearer {token}"},
    )
    if r.status_code != 200:
        _fail(step, r.text, status=r.status_code)
    if "id" not in r.json():
        _fail(step, "respuesta sin id de perfil")
    _ok(step)


def check_predict(client: httpx.Client, base: str, token: str, png: bytes) -> str:
    step = "predict"
    files = {"image": ("smoke.png", png, "image/png")}
    r = client.post(
        f"{base}/predict",
        headers={"Authorization": f"Bearer {token}"},
        files=files,
        timeout=TIMEOUT_PREDICT,
    )
    if r.status_code != 200:
        _fail(step, r.text, status=r.status_code)
    data = r.json()
    risk = data.get("risk")
    if risk not in VALID_RISKS:
        _fail(step, f"risk={risk!r}")
    pred_id = data.get("id")
    if not pred_id:
        _fail(step, "respuesta sin id")
    _ok(step, f"risk={risk} id={pred_id}")
    return str(pred_id)


def check_predictions_list(
    client: httpx.Client,
    base: str,
    token: str,
    *,
    expect_id: str,
) -> None:
    step = "predictions"
    r = client.get(
        f"{base}/predictions",
        headers={"Authorization": f"Bearer {token}"},
        params={"limit": 10},
    )
    if r.status_code != 200:
        _fail(step, r.text, status=r.status_code)
    data = r.json()
    items = data.get("items") or []
    ids = {item.get("id") for item in items}
    if expect_id not in ids:
        _fail(step, f"predicción {expect_id} no aparece en lista ({len(items)} items)")
    _ok(step, f"items={len(items)}")


def check_metrics(client: httpx.Client, base: str, metrics_token: str) -> None:
    step = "metrics"
    r = client.get(
        f"{base}/metrics",
        headers={"Authorization": f"Bearer {metrics_token}"},
    )
    if r.status_code != 200:
        _fail(step, r.text, status=r.status_code)
    if "model_loaded" not in r.text:
        _fail(step, "cuerpo sin métrica model_loaded")
    _ok(step)


def main() -> None:
    base = _base_url()
    email = _env("SMOKE_EMAIL", required=True)
    password = _env("SMOKE_PASSWORD", required=True)
    metrics_token = _env("METRICS_BEARER_TOKEN", required=True)

    print(f"Smoke prod → {base}")

    png = skin_patch_png()
    with httpx.Client(timeout=TIMEOUT_DEFAULT) as client:
        check_health(client, base)
        token = obtain_access_token(client, base, email, password)
        check_profile(client, base, token)
        pred_id = check_predict(client, base, token, png)
        check_predictions_list(client, base, token, expect_id=pred_id)
        check_metrics(client, base, metrics_token)

    print("Smoke prod: todos los pasos OK")


if __name__ == "__main__":
    main()
