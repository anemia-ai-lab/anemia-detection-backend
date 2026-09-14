# Experimentos para conferencia (anemia / uña)

Índice de runs reproducibles en `ml/artifacts/runs/`. Solo los runs de la whitelist en `.gitignore` se versionan en git; el resto son locales (reproducibles con los comandos Makefile).

**Modelo oficial tesis v2 (pediátrico):** ensemble 3 semillas Ghana unique-hash (1997 PNG) + **Platt** + tiers bajo/medio/alto. Ver [`MOBILE_INFERENCE.md`](MOBILE_INFERENCE.md). Recorte de inferencia: `tip_to_dip_rotated`.

**v1 (referencia):** Ghana scratch + augmentación → `baseline_mobilenetv2_ghana_augmented.keras` (AUC test ~0.69 en test con clones).

| run_id / prefijo | Hipótesis | Train | Test | AUC test | Recall @ τ | Artefacto |
|------------------|-----------|-------|------|----------|------------|-----------|
| `experiment_20260601T045054Z` | Nature baseline (ImageNet + cabezal + FT 10) | `data/train` | `data/test` | **0.773** | 0.704 | `experiment_20260601T045054Z` (local) |
| `eval_ghana_external_20260601T045328Z` | Nature + cal. Nature → Ghana (dominio distinto) | — | `ghana/test` | **0.562** | 0.647 | `eval_ghana_external_20260601T045328Z` (local) |
| `calibration_20260601T045820Z` | Transfer fine-tune Nature→Ghana | `ghana/train` | `ghana/test` | **0.557** | 0.275 | `calibration_20260601T045820Z` (local) |
| `experiment_20260601T050706Z` | Ghana scratch (`original-only`, 507 crops) | `ghana/train` | `ghana/test` | **0.468** | 0.725 | `experiment_20260601T050706Z` (local) |
| `experiment_20260601T051158Z` | Ghana scratch + **augmented** (~3960 crops, clones) | `ghana/train` | `ghana/test` | **0.690** | 0.764 | `experiment_20260601T051158Z` (local) |
| `calibration_20260601T052254Z` | Calibración modelo augmented v1 | `ghana/train` | `ghana/test` | **0.690** | — | `calibration_20260601T052254Z` (local) |
| `calibration_ensemble_ghana_v2` (clones, jun 2026) | Ensemble 3× en test con duplicados SHA | `ghana/train` | `ghana/test` | **0.682** | — | histórico hasta sep 2026 |
| `calibration_ensemble_ghana_dedup` | Pesos jun-2026, test **únicos SHA-256** (1997). Platt vs T vs isotonic | `ghana/train` | `ghana/test` | **0.709** | — | [JSON](../artifacts/runs/calibration_ensemble_ghana_dedup.json) |
| `calibration_ensemble_ghana_v2` (sep 2026) | Retrain 3 seeds unique-hash + Platt | `ghana/train` | `ghana/test` | **0.640** | 0.622 | [JSON](../artifacts/runs/calibration_ensemble_ghana_v2.json) |
| `eval_ensemble_ghana_test_augmented_*` | Eval ensemble en test augmented (clones) | — | `ghana/test` | **0.682** | — | `eval_ensemble_ghana_test_augmented_20260601T064622Z` (local) |
| `experiment_*_seed123` / `seed456` | Mismos hiperparámetros, semillas distintas | `ghana/train` | `ghana/test` | ver JSON | — | `baseline_mobilenetv2_ghana_augmented_seed*.keras` |
| `experiment_20260601T064911Z` | Focal loss (γ=2), seed 42 | `ghana/train` | `ghana/test` | **0.675** | 0.696 | `experiment_20260601T064911Z` (local) |
| `experiment_20260601T065733Z` | Aug online tipo móvil | `ghana/train` | `ghana/test` | **0.668** | 0.698 | `experiment_20260601T065733Z` (local) |
| `eval_nature_pediatric_model_*` | Modelo pediátrico + cal. Ghana sobre Nature | — | `data/test` | **0.475** (cal.) | 1.000 | `eval_nature_pediatric_model_20260601T050956Z` (local) |

## Limitaciones (tesis / producto Perú)

- Proxy **Ghana** (niños ≤5 años); **sin** cohorte peruana con Hb ni re-etiquetado por laboratorio.
- Prepare `--include-augmented` + SHA-256: **1997 únicos** (1963 duplicados omitidos; 3960 PNG reconocidos de 4260 raw, 507 sujetos; split train 1603 / test 394, 402+100 sujetos).
- Retrain sep 2026 (Docker no disponible: `ml/.venv`, mismos flags que `make ml-docker-train-ghana-ensemble-seeds`). Ensemble calibrado: Platt ECE **0.0657** vs T **0.0850** vs isotonic **0.0618** (ablación). AUC **0.640**. τ alto **0.578**, low_upper **0.491**. El AUC baja vs 0.682/0.709 de pesos entrenados con clones: el test unique-hash ya no filtra duplicados del train.
- Agregación **median** (no max) mitiga la amplificación del sesgo en inferencia multinail; **no** cierra el domain gap de fotos reales (Etapa 3). Recorte **rotado** tip→DIP (centro 0.35 hacia el lecho); `crop_scale` default **1.0**.
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

Si Docker no corre, el equivalente local es `ml/.venv` con los mismos flags de `train.py` / `calibrate_ensemble_eval.py` / `export_ensemble_tflite.py`.

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
| Ensemble .keras | `baseline_mobilenetv2_ghana_augmented_seed{42,123,456}.keras` (retrain unique-hash 2026-09-14) |
| Calibración + tiers | `calibration_ensemble_ghana_v2.json` (Platt a≈1.467 b≈0.625, T≈0.944, low_upper≈0.491, high_lower≈0.578) |
| Diagnóstico pesos antiguos | `calibration_ensemble_ghana_dedup.json` (Platt ECE 0.088 en test unique-hash **sin** retrain) |
| TFLite móvil | `baseline_mobilenetv2_ghana_augmented_seed*.tflite` + `baseline_mobilenetv2_ghana_ensemble.metadata.json` (`crop: tip_to_dip_rotated`) |
| API | `INFERENCE_MODEL_PATHS` + `INFERENCE_CALIBRATION_METHOD=platt` + `INFERENCE_RISK_TIER_*` en `.env` |

Single-seed v1 (referencia, test con clones): AUC **0.690** — `experiment_20260601T051158Z` / `calibration_20260601T052254Z`.

## Conclusión para slides

- **Transferencia de dominio Nature→Ghana falla** (AUC ~0.56); no usar Nature como init para tesis pediátrica.
- **Ghana solo con crops originales** no mejora (AUC ~0.47).
- **Ghana augmented (v1 single seed, clones)** AUC **~0.69**; **ensemble jun-2026** AUC **~0.68** en test con clones; **0.709** esos mismos pesos en test unique-hash.
- **Ensemble retrain unique-hash (sep 2026)** AUC **0.640**, ECE Platt **0.066** (elige Platt frente a T). Cifra más honesta: sin clones exactos en train/test.
- Focal y aug móvil **no superan** v1 en test augmented (0.675 / 0.668).
- Evaluación en Nature con modelo pediátrico confirma **no** optimización para adultos (AUC cal. ~0.47).
- **Etapa 1+2** (median, min 2 uñas, dedup, Platt, crop rotado) reduce amplificación del sesgo; **no** cierra domain gap (Etapa 3).

## Referencia histórica (no versionada en git)

Runs Nature (abril 2026) y experimentos intermedios de junio se conservan solo en local al reproducir con Makefile. En git: `calibration_ensemble_ghana_v2` (producción unique-hash) y `calibration_ensemble_ghana_dedup` (diagnóstico pesos jun-2026).
