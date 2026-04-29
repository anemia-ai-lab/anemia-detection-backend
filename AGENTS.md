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
	•	`make test` — solo `tests/` con `DISABLE_TF=1` e `INFERENCE_MODEL_PATH` vacío (sin TensorFlow en la suite del API); `make ml-test` — `ml/tests/` con TensorFlow y artefactos locales.
	•	`make lint` / `make format` — ruff.

Supabase keys (security):
	•	`SUPABASE_KEY` with `get_supabase_client()`: subject to RLS; still a server secret — do not ship to browsers.
	•	`SUPABASE_SERVICE_ROLE_KEY` with `get_supabase_service_client()`: bypasses RLS — trusted server only, never expose to clients.