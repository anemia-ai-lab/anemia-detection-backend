# Release 1 — smoke checklist (manual)

Fecha: _rellenar tras deploy v2_

Base URL prod: `https://anemia-detection-backend.onrender.com`

## Pre-deploy

- [ ] `make lint && make test && make ml-test-docker` verde
- [ ] `docker build -f Dockerfile .` exitoso (3× `.keras` en imagen)
- [ ] `supabase db push` — remoto al día
- [ ] Variables en Render según [`render.env.example`](../render.env.example)

## Smoke HTTP

| # | Request | Esperado |
|---|---------|----------|
| 1 | `GET /health` | `status=ok`, `model_loaded=true`, `model_version=v2.0` |
| 2 | `POST /auth/register` | 201/200 + tokens |
| 3 | `POST /auth/login` | 200 + JWT |
| 4 | `GET /auth/me/profile` | 200 con JWT |
| 5 | `POST /predict` | 200, `risk_tier` en low/medium/high |
| 6 | `GET /predictions` | 200 lista paginada |
| 7 | `POST /predictions/sync/metadata` | 200 batch offline |
| 8 | `GET /metrics` + `Authorization: Bearer $METRICS_BEARER_TOKEN` | 200 Prometheus |

## Estado pre-R1 (2026-06-24)

- **Imagen local** (`docker build -f Dockerfile .`): `GET /health` → `status=ok`, `model_loaded=true`, `model_version=v2.0`, ensemble `n=3`.
- **Render prod** (pre-deploy v2): `/health` → `degraded`, `model_loaded=false`, `model_version=v1.0` — actualizar env + redeploy según [`render.env.example`](../render.env.example).
- Supabase migraciones: remoto al día (`supabase db push --dry-run`).

## Limitaciones (release notes)

- Cribado/investigación; no diagnóstico clínico.
- Modelo proxy Ghana pediátrico; sin validación cohorte Perú.
- Rate limit in-memory (una instancia).
