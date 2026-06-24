# Experimentos para conferencia (anemia / uña)

Índice de runs reproducibles en `ml/artifacts/runs/`. Solo los runs de la whitelist en `.gitignore` se versionan en git; el resto son locales (reproducibles con los comandos Makefile).

**Modelo oficial tesis v2 (pediátrico):** ensemble 3 semillas Ghana augmented + calibración + tiers bajo/medio/alto. Ver [`MOBILE_INFERENCE.md`](MOBILE_INFERENCE.md).

**v1 (referencia):** Ghana scratch + augmentación → `baseline_mobilenetv2_ghana_augmented.keras` (AUC test ~0.69).

| run_id / prefijo | Hipótesis | Train | Test | AUC test | Recall @ τ | Artefacto |
|------------------|-----------|-------|------|----------|------------|-----------|
| `experiment_20260601T045054Z` | Nature baseline (ImageNet + cabezal + FT 10) | `data/train` | `data/test` | **0.773** | 0.704 | `experiment_20260601T045054Z` (local) |
| `eval_ghana_external_20260601T045328Z` | Nature + cal. Nature → Ghana (dominio distinto) | — | `ghana/test` | **0.562** | 0.647 | `eval_ghana_external_20260601T045328Z` (local) |
| `calibration_20260601T045820Z` | Transfer fine-tune Nature→Ghana | `ghana/train` | `ghana/test` | **0.557** | 0.275 | `calibration_20260601T045820Z` (local) |
| `experiment_20260601T050706Z` | Ghana scratch (`original-only`, 507 crops) | `ghana/train` | `ghana/test` | **0.468** | 0.725 | `experiment_20260601T050706Z` (local) |
| `experiment_20260601T051158Z` | Ghana scratch + **augmented** (~3960 crops) | `ghana/train` | `ghana/test` | **0.690** | 0.764 | `experiment_20260601T051158Z` (local) |
| `calibration_20260601T052254Z` | Calibración modelo augmented v1 | `ghana/train` | `ghana/test` | **0.690** | — | `calibration_20260601T052254Z` (local) |
| `calibration_ensemble_ghana_v2` | Ensemble 3× seed (mean raw) + tiers | `ghana/train` | `ghana/test` | **0.682** | — | [JSON](../artifacts/runs/calibration_ensemble_ghana_v2.json) |
| `eval_ensemble_ghana_test_augmented_*` | Eval ensemble en test augmented | — | `ghana/test` | **0.682** | — | `eval_ensemble_ghana_test_augmented_20260601T064622Z` (local) |
| `experiment_*_seed123` / `seed456` | Mismos hiperparámetros, semillas distintas | `ghana/train` | `ghana/test` | ver JSON | — | `baseline_mobilenetv2_ghana_augmented_seed*.keras` |
| `experiment_20260601T064911Z` | Focal loss (γ=2), seed 42 | `ghana/train` | `ghana/test` | **0.675** | 0.696 | `experiment_20260601T064911Z` (local) |
| `experiment_20260601T065733Z` | Aug online tipo móvil | `ghana/train` | `ghana/test` | **0.668** | 0.698 | `experiment_20260601T065733Z` (local) |
| `eval_nature_pediatric_model_*` | Modelo pediátrico + cal. Ghana sobre Nature | — | `data/test` | **0.475** (cal.) | 1.000 | `eval_nature_pediatric_model_20260601T050956Z` (local) |

## Limitaciones (tesis / producto Perú)

- Proxy **Ghana** (niños ≤5 años); **sin** cohorte peruana con Hb ni re-etiquetado por laboratorio.
- Mejora en campo depende del **protocolo de foto** (anular, medio, índice) y del **recorte OpenCV**, no solo del CNN.
- Tier **medio** = zona gris; no sustituye hemograma ni diagnóstico.

## Comandos Makefile

```sh
make ml-docker-train-ghana-scratch      # original-only
make ml-docker-prepare-ghana-augmented
make ml-docker-train-ghana-augmented   # recomendado si AUC bajo
make ml-docker-train-ghana-ensemble-seeds   # seeds 42, 123, 456
make ml-docker-calibrate-ensemble-ghana
make ml-docker-export-ensemble-tflite
make ml-docker-train-ghana-focal
make ml-docker-train-ghana-mobile-aug
make ml-docker-calibrate-ghana         # calibrar baseline_mobilenetv2_ghana.keras
```

Tras calibrar el modelo augmented manualmente:

```sh
python ml/scripts/calibrate_eval.py \
  --model-path artifacts/models/baseline_mobilenetv2_ghana_augmented.keras \
  --train-dir data/ghana/train --test-dir data/ghana/test
python ml/scripts/sync_calibration_constants.py --calibration-json ml/artifacts/runs/calibration_<UTC>.json
cd ml && python scripts/export_tflite.py --calibration-json artifacts/runs/calibration_<UTC>.json --overwrite
```

## Congelado v2 pediátrico (proxy Perú)

| Artefacto | Ruta / run |
|-----------|------------|
| Ensemble .keras | `baseline_mobilenetv2_ghana_augmented_seed{42,123,456}.keras` |
| Calibración + tiers | `calibration_ensemble_ghana_v2.json` (T≈1.41, low_upper≈0.32, high_lower≈0.38) |
| TFLite móvil | `baseline_mobilenetv2_ghana_augmented_seed*.tflite` + `baseline_mobilenetv2_ghana_ensemble.metadata.json` |
| API | `INFERENCE_MODEL_PATHS` + `INFERENCE_RISK_TIER_*` en `.env` |

Single-seed v1 (referencia): AUC **0.690** — `experiment_20260601T051158Z` / `calibration_20260601T052254Z`.

## Conclusión para slides

- **Transferencia de dominio Nature→Ghana falla** (AUC ~0.56); no usar Nature como init para tesis pediátrica.
- **Ghana solo con crops originales** no mejora (AUC ~0.47).
- **Ghana augmented (v1 single seed)** AUC **~0.69**; **ensemble v2** AUC **~0.68** (estable, tiers bajo/medio/alto).
- Focal y aug móvil **no superan** v1 en test augmented (0.675 / 0.668).
- Evaluación en Nature con modelo pediátrico confirma **no** optimización para adultos (AUC cal. ~0.47).

## Referencia histórica (no versionada en git)

Runs Nature (abril 2026) y experimentos intermedios de junio se conservan solo en local al reproducir con Makefile. En git queda únicamente `calibration_ensemble_ghana_v2`.
