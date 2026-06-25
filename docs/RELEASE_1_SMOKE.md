# Release 1 — smoke (prod)

Base URL prod: `https://anemia-detection-backend.onrender.com`

## Automático

```bash
export SMOKE_EMAIL=smoke@example.com
export SMOKE_PASSWORD=minimum8chars
export METRICS_BEARER_TOKEN=<mismo que Render>
# opcional: export SMOKE_BASE_URL=https://anemia-detection-backend.onrender.com
make smoke-prod
```

**GitHub Actions** (workflow [`.github/workflows/keepalive.yml`](../.github/workflows/keepalive.yml), job `smoke-prod`):

- Triggers: `schedule` (lun/jue 15:00 UTC) y `workflow_dispatch`
- Secrets del repo: `SMOKE_EMAIL`, `SMOKE_PASSWORD`, `METRICS_BEARER_TOKEN`
- Variable opcional: `SMOKE_BASE_URL` (sin barra final)

El script registra el usuario en el primer run si `login` devuelve 401.

## Pasos que ejecuta `scripts/smoke_prod.py`

| # | Request | Esperado |
|---|---------|----------|
| 1 | `GET /health` | `status=ok`, `model_loaded=true`, `model_version=v2.0` |
| 2 | `POST /auth/login` (o `register` + `login`) | 200 + JWT |
| 3 | `GET /auth/me/profile` | 200 con JWT |
| 4 | `POST /predict` | 200, `risk` en low/medium/high |
| 5 | `GET /predictions` | 200, incluye la predicción del paso 4 |
| 6 | `GET /metrics` + `Authorization: Bearer $METRICS_BEARER_TOKEN` | 200 Prometheus |

## Pre-deploy (CI local)

- [ ] `make lint && make test && make ml-test-docker` verde
- [ ] `docker build -f Dockerfile .` exitoso (3× `.keras` en imagen)
- [ ] `supabase db push` — remoto al día
- [ ] Variables en Render según [`render.env.example`](../render.env.example) (mínimo obligatorio)

## Estado (2026-06-24)

- **Render prod:** `/health` → `status=ok`, `model_loaded=true`, `model_version=v2.0` (ensemble v2).
- **Smoke:** automatizado vía `make smoke-prod` + job `smoke-prod` en keepalive.

## Limitaciones (release notes)

- Cribado/investigación; no diagnóstico clínico.
- Modelo proxy Ghana pediátrico; sin validación cohorte Perú.
- Rate limit in-memory (una instancia).
