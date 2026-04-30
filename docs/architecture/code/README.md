# C4 Code Diagrams

This folder contains curated C4 level 4 (code) diagrams for the FastAPI backend.

The diagrams are generated with a hybrid workflow:

1. Generate raw PlantUML from Python packages with `pyreverse`.
2. Use the generated files as references only.
3. Keep the final `.puml` diagrams curated by hand so they stay readable and aligned with the C3 component model in `docs/architecture/workspace.dsl`.

## Files

- `overview.puml` - package-level backend overview.
- `api.puml` - FastAPI application, routes, dependencies, and middleware.
- `services.puml` - service layer and domain orchestration.
- `repositories.puml` - Supabase-backed persistence and storage adapters.
- `inference.puml` - inference runtime, image predictor contract, Keras implementation, validation, calibration, and risk mapping.
- `core.puml` - configuration, auth token parsing, health/metrics/rate limiting, and utility modules.
- `schemas.puml` - API contract models grouped by feature.

## Generate Raw Diagrams

Run from the repository root:

```sh
make c4-code-generate
```

Raw output is written to `docs/architecture/code/_generated/`. This folder is not versioned.

## Curate Diagrams

Curated diagrams live directly in `docs/architecture/code/`. Do not copy raw `pyreverse` output blindly. Keep only the classes, protocols, modules, and relationships that help explain the code-level structure behind the C3 diagram.

## Render SVGs

Render only when needed:

```sh
make c4-code-render
```

SVG output is written to `docs/architecture/code/rendered/`. This folder is not versioned.

## Clean Generated Files

```sh
make c4-code-clean
```
