# Release v1.0.0 — Backend + ML (software)

**Fecha:** 2026-06-24  
**Modelo interno:** v2.0 ensemble (3× Ghana augmented, tiers low/medium/high)

## Qué incluye

- API FastAPI: auth, perfil, `POST /predict`, historial, sync offline, Storage de imágenes
- Ensemble Keras en servidor (media de `raw_prob` + calibración por temperatura)
- Artefactos móvil: 3× TFLite + `baseline_mobilenetv2_ghana_ensemble.metadata.json`
- Migraciones Supabase versionadas; RLS en `predictions` y `profiles`
- CI: lint + tests API (`DISABLE_TF=1`) + tests ML en Docker Linux

## Limitaciones (obligatorio leer)

- **Cribado e investigación** — no diagnóstico clínico ni recomendación terapéutica.
- Dataset proxy **Ghana pediátrico**; sin validación en cohorte peruana.
- Rate limit **in-memory** (adecuado para demo/piloto con **1** máquina Fly).
- App móvil es cliente externo; contrato offline en [`ml/docs/MOBILE_INFERENCE.md`](../ml/docs/MOBILE_INFERENCE.md).

## Artefactos móvil (en repo)

- `ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed{42,123,456}.tflite`
- `ml/artifacts/models/baseline_mobilenetv2_ghana_ensemble.metadata.json`

## Producción (Fly.io)

Variables y despliegue: [`docs/DEPLOYMENT_FLY.md`](DEPLOYMENT_FLY.md). IaC AWS histórico: [`docs/DEPLOYMENT_AWS.md`](DEPLOYMENT_AWS.md).

## Smoke producción

Base URL prod: variable `SMOKE_BASE_URL` (`https://<app>.fly.dev`, sin barra final).

### Automático

```bash
export SMOKE_EMAIL=smoke@example.com
export SMOKE_PASSWORD=minimum8chars
export METRICS_BEARER_TOKEN=<mismo que fly secrets>
export SMOKE_BASE_URL=https://<app>.fly.dev
make smoke-prod
```

**GitHub Actions**

- **Deploy Fly** ([`.github/workflows/deploy-fly.yml`](../.github/workflows/deploy-fly.yml)): push a `main` (rutas relevantes) o `workflow_dispatch`. Secret `FLY_API_TOKEN`.
- **CI smoke programado** ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml), job `smoke-prod`): `schedule` (lun/jue 15:00 UTC) y `workflow_dispatch`
- Secrets: `SMOKE_EMAIL`, `SMOKE_PASSWORD`, `METRICS_BEARER_TOKEN`
- Variable para CI programado: `SMOKE_BASE_URL` (`https://<app>.fly.dev`, sin barra final)

El script registra el usuario en el primer run si `login` devuelve 401.

### Pasos que ejecuta `scripts/smoke_prod.py`

| # | Request | Esperado |
|---|---------|----------|
| 1 | `GET /health` | `status=ok`, `model_loaded=true`, `model_version=v2.0`, `supabase_ready=true`, `hand_landmarker_ready=true` (si multinail activo) |
| 2 | `POST /auth/login` (o `register` + `login`) | 200 + JWT |
| 3 | `GET /auth/me/profile` | 200 con JWT |
| 4 | `POST /predict` (JPEG `scripts/fixtures/smoke_hand.jpg`, MediaPipe) | 200, `risk` válido, `preprocessing.detector=mediapipe_hands` |
| 5 | `GET /predictions` | 200, incluye la predicción del paso 4 |
| 6 | `POST /predictions/sync/metadata` + `POST /predictions/{id}/image` + idempotencia + `GET /predictions/{id}` | sync offline `tflite_offline`, `has_image=true` |
| 7 | `GET /metrics` + `Authorization: Bearer $METRICS_BEARER_TOKEN` | 200 Prometheus |

## Pre-deploy checklist

- [ ] `make lint && make test && make ml-test-docker` verde
- [ ] `docker build -f Dockerfile .` exitoso (3× `.keras` en imagen)
- [ ] `supabase db push` — remoto al día
- [ ] Secretos en Fly (`fly secrets set`) según [`docs/DEPLOYMENT_FLY.md`](DEPLOYMENT_FLY.md)
