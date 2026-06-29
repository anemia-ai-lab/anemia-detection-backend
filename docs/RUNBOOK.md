# Runbook — reproducción local

Guía operativa (no sustituye la memoria de tesis). **No es uso clínico:** cribado/investigación asistida; **no** diagnóstico, confirmación clínica ni recomendación terapéutica.

## Prerrequisitos

- Python 3.11+ (`make run` usa `python3`; el venv ML usa `python3.11` si existe).
- `pip`; conviene `.venv` en la raíz del repo.
- Proyecto Supabase (Auth, Postgres, Storage).
- Docker recomendado si los tests ML fallan en el host.

## Entorno

1. Raíz del repo; copiar `.env.example` → `.env` ([variables](#variables-de-entorno-críticas)).
2. Dependencias:

   ```bash
   make install
   make download-hand-landmarker
   ```

   `download-hand-landmarker` obtiene `ml/artifacts/models/hand_landmarker.task` (MediaPipe Tasks; requerido si `PREDICT_MULTINAIL_ENABLED=true`). En Docker el `Dockerfile` lo descarga en build.

Levanta variables desde `.env` vía `backend/core/config.py` (`Settings`).

## Supabase y migraciones

1. Proyecto con Auth activado.
2. `SUPABASE_URL`, `SUPABASE_KEY` (anon); `SUPABASE_SERVICE_ROLE_KEY` solo en servidor (nunca en apps cliente).
3. Esquema:

   ```bash
   make db-push
   ```

CLI Supabase enlazada al proyecto. SQL en `supabase/migrations/`. Sin migraciones: la API puede arrancar pero fallará en escrituras según el caso.

## Backend

```bash
make run
```

(`uvicorn backend.main:app --reload`). OpenAPI: `/docs`, `/openapi.json`. Con `INFERENCE_MODEL_PATHS` (producción v2) o `INFERENCE_MODEL_PATH` (legacy) el lifespan carga el modelo; con `DISABLE_TF=1` o rutas vacías puede no cargarse TF.

## Tests API (sin TensorFlow en ese proceso)

```bash
make test
```

El `Makefile` exporta `DISABLE_TF=1` e `INFERENCE_MODEL_PATH=` para que `tests/` no importe TensorFlow. La entrada `DISABLE_TF` en `.env.example` documenta el mismo flag fuera de `make test`.

## Lint

```bash
make lint
```

## Tests ML

```bash
make ml-venv
make ml-install
make ml-test
```

Requiere `ml/.venv` y `ml/requirements.txt`. Si TF en macOS u otro host falla:

```bash
make ml-test-docker
```

(`Dockerfile.ml-test`, Linux reproducible.)

## Pipeline ML (histórico v1 + ensemble v2)

### Producción v2 (ensemble)

Flujo recomendado para el modelo pediátrico en producción (3 semillas, tiers low/medium/high). Detalle en [`ml/README.md`](../ml/README.md).

1. **Ghana augmented:** `make ml-docker-prepare-ghana-augmented` (o preparar datos con `prepare_ghana_dataset.py`).
2. **Entrenar ensemble:** `make ml-docker-train-ghana-ensemble-seeds` (seeds 42, 123, 456).
3. **Calibrar ensemble:** `make ml-docker-calibrate-ensemble-ghana` → `artifacts/runs/calibration_ensemble_ghana_v2.json`.
4. **Export TFLite móvil:** `make ml-docker-export-ensemble-tflite`.
5. **Sync API:** `python ml/scripts/sync_calibration_constants.py --calibration-json ml/artifacts/runs/calibration_ensemble_ghana_v2.json`.

### Pipeline Nature + Ghana (histórico v1)

1. **Nature:** `python ml/scripts/prepare_nature_dataset.py` → `ml/data/train|test`.
2. **Ghana:** copiar PNG a `ml/data_raw/ghana/` → `python ml/scripts/prepare_ghana_dataset.py` → `ml/data/ghana/`.
3. **Entrenar** (Nature): `cd ml && python scripts/train.py --fine-tune-epochs 10 --metadata-path data_raw/nature/metadata.csv`.
4. **Calibrar:** `python scripts/calibrate_eval.py --experiment-json artifacts/runs/experiment_<último>.json`.
5. **Sync API:** `python ml/scripts/sync_calibration_constants.py --calibration-json artifacts/runs/calibration_<último>.json`.
6. **Eval Ghana:** `python scripts/evaluate_dir.py --test-dir data/ghana/test --calibration-json artifacts/runs/calibration_*.json --dataset-label ghana_external`.

**Nota histórica:** el fine-tune Nature→Ghana (`make ml-docker-finetune-ghana`) fue una ablación con AUC ~0,56 — no recomendado para producción. Ver [`ml/docs/CONFERENCE_EXPERIMENTS.md`](../ml/docs/CONFERENCE_EXPERIMENTS.md).

Entrenamiento, calibración y `evaluate_dir` requieren TensorFlow; si `make ml-tf-check` aborta en macOS, usar `make ml-test-docker` o Linux. `prepare_ghana_dataset.py` puede usar `sips` en macOS sin TF.

## Validación (resumen)

| Comando | Uso |
|--------|-----|
| `make test` | API rápida, `DISABLE_TF=1`. |
| `make lint` | Ruff. |
| `make ml-test` | `ml/tests/` + TensorFlow. |
| `make ml-test-docker` | Misma suite ML en contenedor. |

Los tests validan software y artefactos, no validez clínica.

## Smoke HTTP

**Local** (con `.env` válido y `make run`): `GET /health`, `GET /docs`, `POST /auth/register|login`, `POST /predict` (JWT + multipart; 200 con inferencia real solo si el modelo está cargado). No commitear secretos.

**Producción (automático):** `make smoke-prod` contra AWS ALB (`scripts/smoke_prod.py`). Requiere `SMOKE_BASE_URL`, `SMOKE_EMAIL`, `SMOKE_PASSWORD`, `METRICS_BEARER_TOKEN`. CI: job `smoke-prod` en [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). Detalle: [`docs/RELEASE.md`](RELEASE.md).

**AWS (env):** secretos en Secrets Manager; task env según [`aws.env.example`](../aws.env.example). Guía completa: [`docs/DEPLOYMENT_AWS.md`](DEPLOYMENT_AWS.md).

## Paridad API vs offline

- Tensor tras RGB validado: `ml.preprocessing.pipeline` → `backend/inference/keras_image_predictor.py`.
- **Online:** `POST /predict` detecta 3 uñas (`backend/inference/nail_detection.py`, MediaPipe Hand Landmarker Tasks), infiere por uña y agrega con `max` (misma semántica que offline).
- Offline: TFLite + metadatos (`ml/README.md`).
- Cabecera de imagen / límites / decode previo a uña: `backend/inference/prediction_image_input.py`, alineado con el decode documentado en `ml/preprocessing/pipeline.py`. Umbrales y matemática de inferencia: código, no este runbook.

## Fallos habituales

| Síntoma | Causa probable |
|--------|----------------|
| 401/403 | JWT o proyecto Supabase incorrecto. |
| 502 al guardar | PostgREST / RLS / migraciones. |
| 503 en `/predict` | Sin modelo, ruta vacía, o TF desactivado. |
| ML OK en Docker, mal en macOS | Rueda TF del host → `make ml-test-docker`. |
| CORS | Fuera de `development`, definir `CORS_ALLOWED_ORIGINS`. |

## Rendimiento e índices Postgres

Índices actuales (migraciones en `supabase/migrations/`):

| Tabla | Índice | Cubre |
|-------|--------|-------|
| `predictions` | `predictions_user_id_idx` (`user_id`) | Filtro RLS + lookups por usuario. |
| `predictions` | `predictions_user_effective_created_idx` (`user_id`, `effective_created_at` DESC, `id` DESC) | Listado paginado `GET /predictions`. |
| `predictions` | `predictions_user_client_id_uidx` (`user_id`, `client_id`) parcial | Sync offline idempotente. |
| `profiles` | PK (`id`) | `GET/PATCH /auth/me/profile` por `auth.uid()`. |

No hace falta índice adicional mientras el volumen sea bajo/medio. Si el historial crece (>100k filas/usuario), ejecutar **Supabase → Database → Performance Advisor** y revisar latencia de `list_for_user_paginated` con `EXPLAIN ANALYZE`.

Métricas por fase de `POST /predict` en `/metrics`: `predict_phase_duration_seconds{phase="preprocess|inference|storage_upload|db_insert"}`.

### Baseline de latencia (antes de cambios de caché)

Establecer p50/p95 por fase en producción (requiere `METRICS_BEARER_TOKEN`):

```bash
curl -sS -H "Authorization: Bearer $METRICS_BEARER_TOKEN" \
  "$SMOKE_BASE_URL/metrics" | grep predict_phase_duration_seconds
```

Interpretación esperada:

| Fase | Cuello de botella típico |
|------|--------------------------|
| `inference` | CPU (ensemble Keras + MediaPipe) — **no** se mejora con caché HTTP |
| `storage_upload` | Supabase Storage — revisar red/región |
| `db_insert` | PostgREST — revisar índices si crece el historial |
| `preprocess` | Decodificación/redimensionado de imagen |

Registrar el baseline (fecha + valores) antes de desplegar optimizaciones de caché. Si `inference` domina, priorizar `INFERENCE_TTA_ENABLED=false` y CPU de Fargate.

Automático:

```bash
export SMOKE_BASE_URL=http://<LoadBalancerDNS>
export METRICS_BEARER_TOKEN=<token>
make metrics-baseline
```

## Caché in-memory

| Capa | Implementación |
|------|----------------|
| Modelo Keras / MediaPipe | Singleton en memoria del proceso |
| URLs firmadas Storage | `TTLCache` local (~50 min) |
| `GET /model/evaluation` | `Cache-Control: public, max-age=3600` |
| Rate limit | In-memory por proceso (adecuado con **1 réplica ECS**) |
| Auth `get_user` | Caché local corta (`AUTH_USER_CACHE_TTL_SECONDS`, default 60) |

Producción actual: **1 tarea Fargate** (`desiredCount: 1`). Escala horizontal (2+ tareas) queda fuera de alcance: el rate limit no sería global entre instancias.

## Variables de entorno críticas

| Variable | Uso |
|----------|-----|
| `SUPABASE_URL` | URL del proyecto. |
| `SUPABASE_KEY` | Clave anon/public. |
| `SUPABASE_SERVICE_ROLE_KEY` | Solo servidor (bypass RLS en bootstrap). |
| `APP_ENV` / `DEBUG` | Entorno; `APP_ENV=production` exige `SUPABASE_*`, `METRICS_BEARER_TOKEN`, prohíbe `DEBUG=true` y bucket distinto de `prediction-images`. |
| `MODEL_VERSION` | Versión persistida y expuesta en API. |
| `INFERENCE_MODEL_PATHS` | Producción v2: lista CSV de `.keras` (ensemble 3 semillas). |
| `INFERENCE_MODEL_PATH` | Legacy / fallback: un solo `.keras`; vacío = sin modelo cargado. |
| `INFERENCE_CALIBRATION_TEMPERATURE` | Temperatura de calibración post-hoc. |
| `INFERENCE_CALIBRATION_OPERATIONAL_THRESHOLD` | Umbral operacional para `POST /predict` (calibrado). |
| `INFERENCE_RISK_TIER_LOW_UPPER` | Límite superior del tier `low` (v2). |
| `INFERENCE_RISK_TIER_HIGH_LOWER` | Límite inferior del tier `high` (v2). |
| `DISABLE_TF` | Omite carga TF en runtime cuando aplica (p. ej. tests). |
| `METRICS_BEARER_TOKEN` | Protege `/metrics` fuera de entornos locales si está definido. |
| `RATE_LIMIT_*`, `TRUST_PROXY_HEADERS` | Rate limit in-memory y confianza en proxy. |
| `RATE_LIMIT_SYNC_METADATA_REQUESTS` | Límite POST `/predictions/sync/metadata` (default 10/min). |
| `RATE_LIMIT_SYNC_IMAGE_REQUESTS` | Límite POST `/predictions/{id}/image` (default 30/min). |
| `AUTH_USER_CACHE_TTL_SECONDS` | TTL caché local de `get_user` (0 = desactivado; default 60). |
| `SUPABASE_JWT_SECRET` | Opcional: verificación JWT local (Dashboard → JWT Secret). |
| `INFERENCE_TTA_ENABLED` | `false` en prod si p95 de `inference` es alto (TTA duplica inferencias). |
| `PREDICTIONS_STORAGE_BUCKET` | Bucket de imágenes (debe coincidir con migraciones SQL; no cambiar sin nueva migración). |
| `PREDICTION_IMAGE_MAX_BYTES` | Tope de subida (default **20 MB**). |
| `PREDICTION_IMAGE_MAX_PIXELS` | Tope al decodificar (default **50 MP**; fotos iPhone). |
| `PREDICTION_IMAGE_STORAGE_MAX_EDGE_PX` | Lado máx. del PNG guardado en Storage (default 1024); inferencia usa RGB completo. |
| `PREDICT_MULTINAIL_ENABLED` | Si `true`, `POST /predict` detecta 3 uñas y agrega con `max`. |
| `HAND_LANDMARKER_MODEL_PATH` | Ruta al `.task` de Hand Landmarker (defecto: `ml/artifacts/models/hand_landmarker.task`). |
| Docker prod | Debian bookworm: `libgl1-mesa-glx`, `libgles2-mesa`, `libegl1-mesa` en el `Dockerfile`. CI Ubuntu 24.04: `make install-mediapipe-system-libs` (usa `libgl1`, `libglx-mesa0`, `libgles2`, `libegl1`). Rebuild + redeploy ECS tras cambios en Docker. |
| `PREDICT_NAIL_REQUIRE_MEDIAPIPE` | Si `true` (default), sin detección real → **400** `no_fingernail_detected` (no riesgo falso). |
| `PREDICT_NAIL_FALLBACK_MODE` | Default `reject`; `whole`/`vertical_thirds` solo con `PREDICT_NAIL_REQUIRE_MEDIAPIPE=false` (debug). |
| `PREDICT_NAIL_*` | Confianza MediaPipe, mínimo de uñas, escala de recorte. |

Lista ampliada en `.env.example`.
