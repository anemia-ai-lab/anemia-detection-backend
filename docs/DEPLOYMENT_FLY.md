# Despliegue Fly.io (always-on)

Producción actual del API: **Fly.io**, región `sjc` (San José, cercana a Supabase `us-west-2`). Mismo [`Dockerfile`](../Dockerfile) que AWS (FastAPI + TensorFlow CPU + ensemble 3× `.keras`). **Supabase** sigue externo (Auth, Postgres, Storage).

El stack ECS de [`docs/DEPLOYMENT_AWS.md`](DEPLOYMENT_AWS.md) se destruyó; el IaC en `infra/` y el workflow AWS se conservan como referencia, no como runtime vivo.

## Por qué always-on (≥2 vCPU / 4 GB)

TensorFlow + tres Keras + warmup de MediaPipe no admiten **scale-to-zero**. Un cold start dejaría `GET /health` y `POST /predict` fuera de plazo. `fly.toml` fija `auto_stop_machines = "off"`, `min_machines_running = 1`, **2 CPU** y **4096 MB**.

**Una sola máquina.** El rate limit es in-memory; no hay volumen persistente (los `.keras` van en la imagen).

```mermaid
flowchart LR
  mobile[Cliente_movil]
  fly[Fly_sjc_2vCPU_4GB]
  supa[Supabase]
  mobile -->|HTTPS| fly
  fly --> supa
```

## Prerrequisitos

1. [Instalar flyctl](https://fly.io/docs/flyctl/install/) (`brew install flyctl`) y `fly auth login`.
2. Docker no es obligatorio en el portátil: `fly deploy --remote-only` construye en Fly.
3. Los 3 `.keras` oficiales deben existir (el `Dockerfile` los copia).
4. Supabase: migraciones al día (`make db-push`).

## Primer despliegue

```bash
fly auth login
fly apps create anemia-detection-api --org personal   # si el nombre está tomado, editar fly.toml
fly deploy --remote-only
```

Tras el primer `deploy`, **una** máquina (`fly scale count 1`). No subir réplicas.

## Secretos

Mismas claves que Secrets Manager / [`aws.env.example`](../aws.env.example). No van en `fly.toml`.

```bash
fly secrets set \
  SUPABASE_URL="https://YOUR_PROJECT.supabase.co" \
  SUPABASE_KEY="YOUR_ANON_KEY" \
  SUPABASE_SERVICE_ROLE_KEY="YOUR_SERVICE_ROLE_KEY" \
  METRICS_BEARER_TOKEN="GENERATE_STRONG_RANDOM_TOKEN"
```

Opcional: `SUPABASE_JWT_SECRET`, `CORS_ALLOWED_ORIGINS`.

No-secretos (`APP_ENV=production`, `TRUST_PROXY_HEADERS=true`, `INFERENCE_MODEL_PATHS`, multinail) viven en `[env]` de [`fly.toml`](../fly.toml). Calibración T/Platt/τ usa los defaults de `backend/core/config.py` embebidos en la imagen (tras `sync_calibration_constants.py`).

Listar: `fly secrets list`. Rotar: `fly secrets set KEY=nuevo`.

## Smoke

```bash
export SMOKE_EMAIL=smoke@example.com
export SMOKE_PASSWORD=minimum8chars
export METRICS_BEARER_TOKEN=<mismo que fly secrets>
export SMOKE_BASE_URL=https://anemia-detection-api.fly.dev
make smoke-prod
```

Hostname real: `fly status` / `fly apps list`. Sin barra final.

## CI opcional

[`.github/workflows/deploy-fly.yml`](../.github/workflows/deploy-fly.yml): `workflow_dispatch` o push a `main` en rutas del API/imagen. Secret de GitHub: `FLY_API_TOKEN` (`fly tokens create deploy`).

## Operación

| Comando | Uso |
|---------|-----|
| `fly status` | Máquinas y health |
| `fly logs` | Logs del proceso |
| `fly ssh console` | Shell en la VM |
| `fly scale show` | CPU/RAM/count |

Health interno: `GET /health` con gracia **180 s** (carga de TF).

## Costos (orden de magnitud)

Shared 2 CPU / 4 GB always-on es el equivalente al Fargate 2 vCPU / 4 GB. Revisar [pricing Fly](https://fly.io/docs/about/pricing/). No hay ALB aparte; HTTPS termina en Fly.

## Fallos habituales

| Síntoma | Qué revisar |
|---------|-------------|
| Health check timeout al deploy | Gracia 180 s; RAM 4 GB; los 3 `.keras` en el build |
| 503 al arrancar | Secretos `SUPABASE_*` / `METRICS_BEARER_TOKEN`; `APP_ENV=production` |
| Rate limit raro entre requests | Más de una máquina (`fly scale count` debe ser 1) |
| Cold start de minutos | `auto_stop_machines` no debe ser `stop`/`suspend` |
