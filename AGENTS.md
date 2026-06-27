Project: FastAPI Backend

Context:
	•	Backend API for anemia risk detection
	•	Built with FastAPI
	•	Uses Supabase for authentication, database, and storage
	•	Architecture: modular monolith

Guidelines:
	•	Do not add new dependencies without approval
	•	Do not place business logic inside routers
	•	Use services layer for business logic
	•	Prioritize clarity over automation
	•	Keep code simple and maintainable
	•	Do not break existing endpoints

Supabase (official documentation — source of truth for APIs and client behavior):
	•	https://supabase.com/docs
	•	Python client (supabase-py): https://supabase.com/docs/reference/python/introduction
	•	Python initializing: https://supabase.com/docs/reference/python/initializing
	•	Auth: https://supabase.com/docs/guides/auth
	•	Database / Postgres: https://supabase.com/docs/guides/database
	•	Row Level Security: https://supabase.com/docs/guides/database/postgres/row-level-security
	•	Storage: https://supabase.com/docs/guides/storage
	•	REST API (PostgREST): https://supabase.com/docs/guides/api
	•	Client wiring in this repo: backend/integrations/supabase_client.py

Repository layout (modular monolith):
	•	`backend/api/` — FastAPI app and HTTP routes (keep routers thin).
	•	`backend/services/` — business logic.
	•	`backend/integrations/` — Supabase and other external clients.
	•	`backend/core/` — configuration (`Settings` in `backend/core/config.py`).
	•	`backend/repositories/` — data access when persistence grows beyond direct client calls.
	•	`backend/schemas/` — Pydantic models for API contracts.
	•	App entrypoint: `backend/main.py` → `backend.api.app:app`.

Local development:
	•	Environment: copy `.env.example` to `.env` at the repo root; `Settings` reads that file.
	•	`make run` — dev server (uvicorn `--reload`).
	•	`make test` — solo `tests/` con `DISABLE_TF=1` e `INFERENCE_MODEL_PATH` vacío (sin TensorFlow en la suite del API); `make ml-test` — `pytest ml/tests/` con `ml/.venv` y `PYTHONPATH=.`.
	•	`make ml-venv` / `make ml-install` — crea `ml/.venv` (prefiere `python3.11`) e instala `ml/requirements.txt` (TensorFlow 2.19.1 fijado para estabilidad en macOS arm64).
	•	`make ml-tf-check` — verifica que TensorFlow importa en el venv ML; `make ml-test-docker` — misma suite en contenedor Linux (`Dockerfile.ml-test`).
	•	`make lint` / `make format` — ruff.

Supabase keys (security):
	•	`SUPABASE_KEY` (anon): use `create_supabase_anon_client()` for GoTrue and `create_supabase_user_client(jwt)` for PostgREST/Storage with RLS. Do not use a shared singleton for concurrent auth.
	•	`SUPABASE_SERVICE_ROLE_KEY` with `get_supabase_service_client()`: bypasses RLS — trusted server only, never expose to clients.
	•	`PREDICTIONS_STORAGE_BUCKET` must stay `prediction-images` unless Storage RLS migrations are updated.
	•	`RISK_THRESHOLD` is legacy; `POST /predict` uses `INFERENCE_CALIBRATION_OPERATIONAL_THRESHOLD` on calibrated probabilities.
	•	`APP_ENV=production` requires `SUPABASE_*`, `METRICS_BEARER_TOKEN`, and forbids `DEBUG=true`.

AWS deployment:
	•	Production: ECS Fargate + ALB in `us-west-2`, IaC in `infra/` (CDK Python).
	•	Guide: `docs/DEPLOYMENT_AWS.md` · env reference: `aws.env.example`.
	•	Secrets: AWS Secrets Manager `anemia-api/prod` (not in git).
	•	CI deploy: `.github/workflows/deploy-aws.yml` (manual dispatch).