# ML artifacts and code

- Source: `baseline/`, `preprocessing/`, `scripts/`, `inference/`, `explainability/`.
- Server model: `artifacts/models/baseline_mobilenetv2.keras`.
- Offline: `artifacts/models/baseline_mobilenetv2_v1.tflite`, `baseline_mobilenetv2_v1.metadata.json`.
- Published reports: `artifacts/runs/experiment_20260420T043804Z.*`, `calibration_20260420T045056Z.*`.

Git ignores raw data, patient records, local MLflow, ad hoc metrics, Grad-CAM dumps, and scratch outputs.

## Datasets (raw → procesado)

| Dataset | Raw (gitignored) | Script | Procesado |
|---------|------------------|--------|-----------|
| Nature | `data_raw/nature/images/` + `metadata.csv` | `scripts/prepare_nature_dataset.py` | `data/train`, `data/test` |
| Ghana (pediátrico) | `data_raw/ghana/*.png` (4260 con aug.) | `scripts/prepare_ghana_dataset.py` | `data/ghana/train`, `data/ghana/test` |

Ghana: prefijos `Anemic-FN-*` → `positive`, `Non-Anrmic-FN-*` → `negative` (typos del autor). Por defecto `--original-only` (~507 sujetos). En macOS, si TensorFlow aborta al importar, el preparador usa `sips` para redimensionar.

```sh
python ml/scripts/prepare_ghana_dataset.py \
  --input-root ml/data_raw/ghana \
  --output-dir ml/data/ghana
```

**Modelo pediátrico v2 (tesis / Perú):** ensemble 3 semillas + tiers bajo/medio/alto. Contrato móvil: [`docs/MOBILE_INFERENCE.md`](docs/MOBILE_INFERENCE.md).

```sh
make ml-docker-prepare-ghana-augmented
make ml-docker-train-ghana-ensemble-seeds   # seeds 42, 123, 456
make ml-docker-calibrate-ensemble-ghana       # → calibration_ensemble_ghana_v2.json (renombrar último run)
make ml-docker-export-ensemble-tflite
python ml/scripts/sync_calibration_constants.py --calibration-json ml/artifacts/runs/calibration_ensemble_ghana_v2.json
```

**v1 (referencia):** Ghana augmented single seed:

```sh
make ml-docker-train-ghana-augmented
make ml-docker-calibrate-ghana-augmented-tiers
```

Ablaciones: `make ml-docker-train-ghana-focal`, `make ml-docker-train-ghana-mobile-aug`.

**Limitaciones:** proxy Ghana (sin cohorte peruana); tier medio = zona gris; calidad de recorte OpenCV y protocolo anular/medio/índice dominan en campo.

Tabla comparativa: [`docs/CONFERENCE_EXPERIMENTS.md`](docs/CONFERENCE_EXPERIMENTS.md).

Evaluación externa (modelo + calibración Nature, test Ghana):

```sh
cd ml
python scripts/evaluate_dir.py \
  --test-dir data/ghana/test \
  --calibration-json artifacts/runs/calibration_20260420T045056Z.json \
  --dataset-label ghana_external
```

## Reproducibility

Reports note the TF/Keras versions used when they were produced. `ml/requirements.txt` pins TensorFlow 2.19.1 (macOS arm64 + Linux Docker). Run `experiment_20260420T042800Z` is referenced for context only and is not part of the public artifact set.

Export TFLite:

```sh
cd ml
python scripts/export_tflite.py --overwrite
```
