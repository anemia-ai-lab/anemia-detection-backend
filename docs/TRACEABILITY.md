# Trazabilidad (tesis ↔ repositorio)

Mapeo breve de responsabilidades a rutas en el repo (defensa / auditoría).

**Alcance:** cribado de riesgo e investigación asistida por software — **no** diagnóstico médico, confirmación clínica ni recomendación de tratamiento.

## Matriz

| Responsabilidad | Evidencia |
|-----------------|-----------|
| API HTTP (auth, perfil, predicción, historial, URL firmada, salud, métricas, evaluación estática) | `backend/api/routes/`, `backend/api/app.py` |
| Lógica de aplicación | `backend/services/` |
| Supabase (JWT + políticas) | `backend/repositories/`, `backend/integrations/supabase_client.py` |
| Inferencia, imagen, heurística uña | `backend/inference/` |
| ML entrenamiento / export / G9 | `ml/` |
| Postgres, RLS, Storage | `supabase/migrations/` |
| Contratos API | `backend/schemas/` |
| Config y límites | `backend/core/config.py`, `.env.example` |
| Tests API vs ML | `tests/`, `ml/tests/`, `Makefile` |
| C4 y diagramas código | `docs/architecture/` (`workspace.dsl`, `docs/architecture/code/*.puml`) |
| Runbook y README | `docs/RUNBOOK.md`, `README.md` |

## Carpetas

| Ruta | Contenido |
|------|-----------|
| `backend/api/routes/` | Routers `/auth`, `/predict`, `/predictions`, `/model/evaluation`. |
| `backend/services/` | `PredictionService`, `AuthService`, `ProfileService`, `ModelEvaluationService`. |
| `backend/repositories/` | Predicciones, perfiles, storage. |
| `backend/inference/` | Keras runtime, `prediction_image_input`, utilidades. |
| `ml/` | Train/eval/export, `ml/preprocessing/pipeline.py`. |
| `supabase/migrations/` | DDL y RLS. |
| `tests/` | Suite `make test` sin TF obligatorio. |
| `Makefile` | `test`, `lint`, `ml-test`, `ml-test-docker`, `run`, `db-push`. |

## Paridad runtime / offline

| Tema | Dónde |
|------|--------|
| Tensor G9 | `ml/preprocessing/pipeline.py`, `backend/inference/keras_image_predictor.py` |
| Imagen en API | `backend/inference/prediction_image_input.py` (coherente con decode ML; ver código) |
| Calibración / TFLite | `backend/core/config.py`, `ml/README.md` |
| Versión | `MODEL_VERSION` / campos `model_version` |

La documentación no cambia fórmulas de inferencia; solo enlaza responsabilidades.

## Comandos de validación

| Comando | Notas |
|---------|--------|
| `make test` | `DISABLE_TF=1`, suite `tests/`. |
| `make lint` | Ruff. |
| `make ml-test` | TF + `ml/.venv`. |
| `make ml-test-docker` | Si TF local falla. |

## Referencias

1. `README.md` — alcance y límites.
2. `docs/RUNBOOK.md` — instalación.
3. `docs/architecture/workspace.dsl` — C3.
4. `AGENTS.md` — layout del monolito y Supabase.
