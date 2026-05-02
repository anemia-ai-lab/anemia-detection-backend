# C4 code diagrams (level 4)

Curated `.puml` for the FastAPI backend. Optional workflow: `make c4-code-generate` → raw PlantUML under `_generated/` (gitignored) → edit diagrams by hand against `docs/architecture/workspace.dsl`; do not paste raw `pyreverse` blindly.

## Files

- `overview.puml` — packages
- `api.puml` — app, routes, deps, middleware
- `services.puml` — services
- `repositories.puml` — persistence / storage
- `inference.puml` — runtime, predictor, validation
- `core.puml` — config, auth token, health, metrics, rate limit
- `schemas.puml` — Pydantic contracts

## Targets

```sh
make c4-code-generate   # raw → _generated/
make c4-code-render     # SVG → rendered/ (gitignored)
make c4-code-clean
```
