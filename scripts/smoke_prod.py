#!/usr/bin/env python3
"""
Smoke E2E contra el API desplegado (AWS ALB u otro host).

Requiere variables de entorno:
  SMOKE_BASE_URL          — base sin barra final (obligatorio)
  SMOKE_EMAIL / SMOKE_PASSWORD — usuario Supabase dedicado a smoke
  METRICS_BEARER_TOKEN    — token para GET /metrics

Salida: 0 si todos los pasos pasan; 1 en caso contrario.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

DEFAULT_BASE_URL = ""
VALID_RISKS = frozenset({"low", "medium", "high"})
TIMEOUT_DEFAULT = httpx.Timeout(30.0, read=120.0)
TIMEOUT_PREDICT = httpx.Timeout(30.0, read=180.0)
_SMOKE_HAND_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "smoke_hand.jpg"
_DEBUG_LOG_PATH = Path(__file__).resolve().parents[1] / ".cursor" / "debug-45fd61.log"


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": "45fd61",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass
    # #endregion


def _env(name: str, *, required: bool = False, default: str = "") -> str:
    value = os.environ.get(name, default).strip()
    if required and not value:
        print(f"ERROR: falta variable de entorno {name}", file=sys.stderr)
        sys.exit(1)
    return value


def is_retired_render_url(base: str) -> bool:
    raw = base.strip().rstrip("/")
    if not raw:
        return False
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    host = (parsed.hostname or "").lower()
    return host == "onrender.com" or host.endswith(".onrender.com")


def _reject_retired_render_url(base: str) -> None:
    if is_retired_render_url(base):
        print(
            "ERROR: SMOKE_BASE_URL apunta a Render (retirado). "
            "Usa el DNS del ALB AWS, p. ej. "
            "http://<LoadBalancerDNS> (output de CloudFormation AnemiaApiStack).",
            file=sys.stderr,
        )
        sys.exit(1)


def _base_url() -> str:
    raw = _env("SMOKE_BASE_URL", required=True, default=DEFAULT_BASE_URL)
    base = raw.rstrip("/")
    _reject_retired_render_url(base)
    return base


def smoke_hand_jpeg() -> bytes:
    """JPEG con mano real (índice/medio/anular) para MediaPipe en smoke prod."""
    if not _SMOKE_HAND_FIXTURE.is_file():
        print(
            f"ERROR: falta fixture {_SMOKE_HAND_FIXTURE} (imagen de mano para smoke /predict).",
            file=sys.stderr,
        )
        sys.exit(1)
    return _SMOKE_HAND_FIXTURE.read_bytes()


def _fail(step: str, detail: str, *, status: int | None = None) -> None:
    prefix = f"FAIL [{step}]"
    if status is not None:
        prefix += f" HTTP {status}"
    print(f"{prefix}: {detail}", file=sys.stderr)
    sys.exit(1)


def _ok(step: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"OK   [{step}]{suffix}")


def preflight_local_hand_detection(image_bytes: bytes) -> dict[str, object]:
    """Diagnóstico local (runner CI) antes de POST /predict remoto."""
    from io import BytesIO

    result: dict[str, object] = {
        "image_bytes": len(image_bytes),
        "fixture": str(_SMOKE_HAND_FIXTURE),
    }
    try:
        import numpy as np
        from PIL import Image

        from backend.inference.nail_detection import (
            build_nail_detector,
            probe_hand_landmarker_status,
            resolved_hand_landmarker_model_path,
        )

        rgb = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
        result["image_shape"] = list(rgb.shape)
        result["probe"] = probe_hand_landmarker_status(warmup=True)
        result["model_path"] = str(resolved_hand_landmarker_model_path())
        crops = build_nail_detector().detect(rgb)
        result["crop_count"] = len(crops)
        result["crop_sources"] = [c.source for c in crops]
        result["crop_fingers"] = [c.finger for c in crops]
    except Exception as exc:
        result["preflight_error"] = f"{type(exc).__name__}: {exc}"
    return result


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
    if data.get("supabase_ready") is False:
        _fail(
            step,
            "supabase_ready=false — edita Secrets Manager anemia-api/prod (SUPABASE_*) "
            "y fuerza redeploy ECS",
        )
    lm_ready = data.get("hand_landmarker_ready")
    if lm_ready is False:
        lm_err = data.get("hand_landmarker_error") or "unknown"
        _agent_debug_log(
            "H2",
            "smoke_prod.py:check_health",
            "prod_landmarker_not_ready",
            {"hand_landmarker_error": lm_err},
        )
        _fail(step, f"hand_landmarker_ready=false error={lm_err!r}")
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


def check_predict(client: httpx.Client, base: str, token: str, image_bytes: bytes) -> str:
    step = "predict"
    preflight = preflight_local_hand_detection(image_bytes)
    print(f"INFO [predict/preflight] {json.dumps(preflight, ensure_ascii=False)}")
    _agent_debug_log("H1", "smoke_prod.py:check_predict", "local_preflight", preflight)
    files = {"image": ("smoke_hand.jpg", image_bytes, "image/jpeg")}
    r = client.post(
        f"{base}/predict",
        headers={"Authorization": f"Bearer {token}"},
        files=files,
        timeout=TIMEOUT_PREDICT,
    )
    if r.status_code != 200:
        _agent_debug_log(
            "A",
            "smoke_prod.py:check_predict",
            "predict_failed",
            {
                "status": r.status_code,
                "body_preview": r.text[:500],
                "image_bytes": len(image_bytes),
                "fixture": str(_SMOKE_HAND_FIXTURE),
                "preflight": preflight,
            },
        )
        _fail(step, r.text, status=r.status_code)
    body = r.json()
    prep = body.get("preprocessing") or {}
    detector = prep.get("detector")
    nails = prep.get("nails") or []
    if detector != "mediapipe_hands":
        _agent_debug_log(
            "B",
            "smoke_prod.py:check_predict",
            "unexpected_detector",
            {"detector": detector, "nail_count": len(nails)},
        )
        _fail(step, f"detector={detector!r} (expected mediapipe_hands)")
    if len(nails) < 1:
        _fail(step, f"nails={len(nails)} (expected >= 1)")
    risk = body.get("risk")
    if risk not in VALID_RISKS:
        _fail(step, f"risk={risk!r}")
    pred_id = body.get("id")
    if not pred_id:
        _fail(step, "respuesta sin id")
    _agent_debug_log(
        "C",
        "smoke_prod.py:check_predict",
        "predict_ok",
        {
            "detector": detector,
            "nail_count": len(nails),
            "risk": risk,
            "runId": "post-fix",
        },
    )
    _ok(step, f"risk={risk} detector={detector} id={pred_id}")
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


def _offline_sync_metadata_payload(client_id: str) -> dict:
    return {
        "items": [
            {
                "client_id": client_id,
                "risk": "low",
                "score": 0.12,
                "raw_probability": 0.15,
                "calibrated_probability": 0.12,
                "threshold_used": 0.3815443834698594,
                "prediction": 0,
                "model_version": "v2.0",
                "inference_mode": "tflite_offline",
                "client_created_at": "2026-04-30T08:00:00+00:00",
                "preprocessing": {"aggregation": "max", "fingers": ["index", "middle", "ring"]},
            }
        ]
    }


def check_offline_sync(
    client: httpx.Client,
    base: str,
    token: str,
    image_bytes: bytes,
) -> None:
    import uuid

    client_id = str(uuid.uuid4())
    step = "offline-sync/metadata"
    r = client.post(
        f"{base}/predictions/sync/metadata",
        headers={"Authorization": f"Bearer {token}"},
        json=_offline_sync_metadata_payload(client_id),
    )
    if r.status_code != 200:
        _fail(step, r.text, status=r.status_code)
    result = r.json()["results"][0]
    if not result.get("created"):
        _fail(step, "expected created=true on first sync")
    if not result.get("image_pending"):
        _fail(step, "expected image_pending=true")
    pred_id = result.get("id")
    if not pred_id:
        _fail(step, "missing prediction id")
    _ok(step, f"id={pred_id}")

    step = "offline-sync/image"
    files = {"image": ("smoke_hand.jpg", image_bytes, "image/jpeg")}
    r_img = client.post(
        f"{base}/predictions/{pred_id}/image",
        headers={"Authorization": f"Bearer {token}"},
        files=files,
        timeout=TIMEOUT_PREDICT,
    )
    if r_img.status_code != 200:
        _fail(step, r_img.text, status=r_img.status_code)
    if not r_img.json().get("image_signed_url"):
        _fail(step, "missing image_signed_url")
    _ok(step)

    step = "offline-sync/idempotent"
    r_retry = client.post(
        f"{base}/predictions/sync/metadata",
        headers={"Authorization": f"Bearer {token}"},
        json=_offline_sync_metadata_payload(client_id),
    )
    if r_retry.status_code != 200:
        _fail(step, r_retry.text, status=r_retry.status_code)
    retry = r_retry.json()["results"][0]
    if retry.get("created"):
        _fail(step, "expected created=false on retry")
    _ok(step)

    step = "offline-sync/detail"
    r_detail = client.get(
        f"{base}/predictions/{pred_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    if r_detail.status_code != 200:
        _fail(step, r_detail.text, status=r_detail.status_code)
    detail = r_detail.json()
    if detail.get("inference_mode") != "tflite_offline":
        _fail(step, f"inference_mode={detail.get('inference_mode')!r}")
    if not detail.get("has_image"):
        _fail(step, "has_image=false")
    _ok(step)


def main() -> None:
    base = _base_url()
    email = _env("SMOKE_EMAIL", required=True)
    password = _env("SMOKE_PASSWORD", required=True)
    metrics_token = _env("METRICS_BEARER_TOKEN", required=True)

    print(f"Smoke prod → {base}")

    image_bytes = smoke_hand_jpeg()
    with httpx.Client(timeout=TIMEOUT_DEFAULT) as client:
        check_health(client, base)
        token = obtain_access_token(client, base, email, password)
        check_profile(client, base, token)
        pred_id = check_predict(client, base, token, image_bytes)
        check_predictions_list(client, base, token, expect_id=pred_id)
        check_offline_sync(client, base, token, image_bytes)
        check_metrics(client, base, metrics_token)

    print("Smoke prod: todos los pasos OK")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--check-render-url":
        sys.exit(1 if is_retired_render_url(sys.argv[2]) else 0)
    main()
