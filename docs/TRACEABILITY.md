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
| Runbook, release y despliegue | `docs/RUNBOOK.md`, `docs/RELEASE.md`, `docs/DEPLOYMENT_FLY.md`, `fly.toml` |

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
| `fly.toml` | App Fly.io always-on (2 vCPU / 4 GB). |

## Paridad runtime / offline

| Tema | Dónde |
|------|--------|
| Tensor G9 | `ml/preprocessing/pipeline.py`, `backend/inference/keras_image_predictor.py` |
| Imagen en API | `backend/inference/prediction_image_input.py` (coherente con decode ML; ver código) |
| Calibración / TFLite | `backend/core/config.py`, `ml/README.md` |
| Versión | `MODEL_VERSION` / campos `model_version` |
| Sync offline | `backend/services/prediction_service.py`, `ml/docs/MOBILE_INFERENCE.md` |

## Validación cohorte peruana (investigación futura)

El modelo v2 está calibrado en cohorte **Ghana pediátrica**. Antes de uso clínico ampliado en Perú:

- [ ] Dataset etiquetado con hemoglobina en cohorte local
- [ ] Re-calibración de tiers (`low_upper`, `high_lower`) y calibración (T o Platt)
- [ ] Informe de métricas (AUC, sensibilidad operacional, ECE) vs Ghana
- [ ] Actualizar `MODEL_VERSION` y metadatos TFLite móvil

Sin esto, el backend sigue operativo pero la **generalización clínica no está validada** (ver [`docs/RELEASE.md`](RELEASE.md)).

La documentación no cambia fórmulas de inferencia; solo enlaza responsabilidades.

## Comandos de validación

Ver [`docs/RUNBOOK.md`](RUNBOOK.md) §Validación para la tabla de comandos `make test`, `make lint`, `make ml-test` y `make ml-test-docker`.

## Referencias

1. `README.md` — alcance y límites.
2. `docs/RUNBOOK.md` — instalación.
3. `docs/architecture/workspace.dsl` — C3.
4. `AGENTS.md` — layout del monolito y Supabase.
