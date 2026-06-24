# Release v1.0.0 — Backend + ML (software)

**Fecha:** 2026-06-24  
**Modelo interno:** v2.0 ensemble (3× Ghana augmented, tiers low/medium/high)

## Qué incluye

- API FastAPI: auth, perfil, `POST /predict`, historial, sync offline, Storage de imágenes
- Ensemble Keras en servidor (media de `raw_prob` + calibración por temperatura)
- Artefactos móvil: 3× TFLite + `baseline_mobilenetv2_ghana_ensemble.metadata.json`
- Migraciones Supabase versionadas; RLS en `predictions` y `profiles`
- CI: lint + tests API (`DISABLE_TF=1`) + tests ML en Docker Linux

## Verificación local

```bash
make lint && make test && make ml-test-docker
docker build -f Dockerfile .
```

## Producción (Render)

Variables: ver [`render.env.example`](../render.env.example). Smoke manual: [`docs/RELEASE_1_SMOKE.md`](RELEASE_1_SMOKE.md).

## Limitaciones (obligatorio leer)

- **Cribado e investigación** — no diagnóstico clínico ni recomendación terapéutica.
- Dataset proxy **Ghana pediátrico**; sin validación en cohorte peruana.
- Rate limit **in-memory** (adecuado para demo/piloto; no multi-réplica sin Redis).
- App móvil es cliente externo; contrato offline en [`ml/docs/MOBILE_INFERENCE.md`](../ml/docs/MOBILE_INFERENCE.md).

## Artefactos móvil (en repo)

- `ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed{42,123,456}.tflite`
- `ml/artifacts/models/baseline_mobilenetv2_ghana_ensemble.metadata.json`
