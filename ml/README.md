# ML artifacts and code

- Source: `baseline/`, `preprocessing/`, `scripts/`, `inference/`, `explainability/`.
- Server model: `artifacts/models/baseline_mobilenetv2.keras`.
- Offline: `artifacts/models/baseline_mobilenetv2_v1.tflite`, `baseline_mobilenetv2_v1.metadata.json`.
- Published reports: `artifacts/runs/experiment_20260420T043804Z.*`, `calibration_20260420T045056Z.*`.

Git ignores raw data, patient records, local MLflow, ad hoc metrics, Grad-CAM dumps, and scratch outputs.

## Reproducibility

Reports note the TF/Keras versions used when they were produced. `ml/requirements.txt` pins TensorFlow 2.19.1 (macOS arm64 + Linux Docker). Run `experiment_20260420T042800Z` is referenced for context only and is not part of the public artifact set.

Export TFLite:

```sh
cd ml
python scripts/export_tflite.py --overwrite
```
