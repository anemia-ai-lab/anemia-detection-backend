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
- Rate limit **in-memory** (adecuado para demo/piloto; no multi-réplica sin Redis).
- App móvil es cliente externo; contrato offline en [`ml/docs/MOBILE_INFERENCE.md`](../ml/docs/MOBILE_INFERENCE.md).

## Artefactos móvil (en repo)

- `ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed{42,123,456}.tflite`
- `ml/artifacts/models/baseline_mobilenetv2_ghana_ensemble.metadata.json`

## Producción (AWS ECS)

Variables y despliegue: [`docs/DEPLOYMENT_AWS.md`](DEPLOYMENT_AWS.md).

## Smoke producción

Base URL prod: variable `SMOKE_BASE_URL` (DNS del ALB AWS, p. ej. `http://anemia-api-xxx.us-west-2.elb.amazonaws.com`).

### Automático

```bash
export SMOKE_EMAIL=smoke@example.com
export SMOKE_PASSWORD=minimum8chars
export METRICS_BEARER_TOKEN=<mismo que Secrets Manager>
export SMOKE_BASE_URL=http://<LoadBalancerDNS>
make smoke-prod
```

**GitHub Actions** (workflow [`.github/workflows/ci.yml`](../.github/workflows/ci.yml), job `smoke-prod`):

- Triggers: `schedule` (lun/jue 15:00 UTC) y `workflow_dispatch`
- Secrets: `SMOKE_EMAIL`, `SMOKE_PASSWORD`, `METRICS_BEARER_TOKEN`
- Variable obligatoria para smoke: `SMOKE_BASE_URL` (sin barra final)

El script registra el usuario en el primer run si `login` devuelve 401.

### Pasos que ejecuta `scripts/smoke_prod.py`

| # | Request | Esperado |
|---|---------|----------|
| 1 | `GET /health` | `status=ok`, `model_loaded=true`, `model_version=v2.0` |
| 2 | `POST /auth/login` (o `register` + `login`) | 200 + JWT |
| 3 | `GET /auth/me/profile` | 200 con JWT |
| 4 | `POST /predict` | 200, `risk` en low/medium/high |
| 5 | `GET /predictions` | 200, incluye la predicción del paso 4 |
| 6 | `GET /metrics` + `Authorization: Bearer $METRICS_BEARER_TOKEN` | 200 Prometheus |

## Pre-deploy checklist

- [ ] `make lint && make test && make ml-test-docker` verde
- [ ] `docker build -f Dockerfile .` exitoso (3× `.keras` en imagen)
- [ ] `supabase db push` — remoto al día
- [ ] Secretos en AWS Secrets Manager según [`docs/DEPLOYMENT_AWS.md`](DEPLOYMENT_AWS.md)
