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
   ```

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

(`uvicorn backend.main:app --reload`). OpenAPI: `/docs`, `/openapi.json`. Con `INFERENCE_MODEL_PATH` el lifespan carga el modelo; con `DISABLE_TF=1` o ruta vacía puede no cargarse TF.

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

## Validación (resumen)

| Comando | Uso |
|--------|-----|
| `make test` | API rápida, `DISABLE_TF=1`. |
| `make lint` | Ruff. |
| `make ml-test` | `ml/tests/` + TensorFlow. |
| `make ml-test-docker` | Misma suite ML en contenedor. |

Los tests validan software y artefactos, no validez clínica.

## Smoke HTTP

Con `.env` válido y `make run`: `GET /health`, `GET /docs`, `POST /auth/register|login`, `POST /predict` (JWT + multipart; 200 con inferencia real solo si el modelo está cargado). No commitear secretos.

## Paridad API vs offline

- Tensor tras RGB validado: `ml.preprocessing.pipeline` → `backend/inference/keras_image_predictor.py`.
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

## Variables de entorno críticas

| Variable | Uso |
|----------|-----|
| `SUPABASE_URL` | URL del proyecto. |
| `SUPABASE_KEY` | Clave anon/public. |
| `SUPABASE_SERVICE_ROLE_KEY` | Solo servidor (bypass RLS en bootstrap). |
| `APP_ENV` / `DEBUG` | Entorno; `APP_ENV=production` no admite `DEBUG=true` en `Settings`. |
| `MODEL_VERSION` | Versión persistida y expuesta en API. |
| `INFERENCE_MODEL_PATH` | `.keras`; vacío = sin modelo cargado en runtime integrado. |
| `INFERENCE_CALIBRATION_*` | Calibración y umbral operacional. |
| `DISABLE_TF` | Omite carga TF en runtime cuando aplica (p. ej. tests). |
| `METRICS_BEARER_TOKEN` | Protege `/metrics` fuera de entornos locales si está definido. |
| `PREDICTION_IMAGE_*` | Límites de imagen. |
| `RATE_LIMIT_*`, `TRUST_PROXY_HEADERS` | Rate limit y confianza en proxy. |
| `PREDICTIONS_STORAGE_BUCKET` | Bucket de imágenes. |

Lista ampliada en `.env.example`.
