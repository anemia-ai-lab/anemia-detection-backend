# Machine Learning Artifacts

This folder contains the reproducible ML code path for the research prototype:

- `baseline/`, `preprocessing/`, `scripts/`, `inference/`, and `explainability/` contain source code.
- `artifacts/models/baseline_mobilenetv2.keras` is the server-side Keras model used by the backend.
- `artifacts/models/baseline_mobilenetv2_v1.tflite` and `baseline_mobilenetv2_v1.metadata.json` are the offline TensorFlow Lite export and post-processing metadata.
- `artifacts/runs/experiment_20260420T043804Z.*` and `calibration_20260420T045056Z.*` are the final public experiment and calibration reports.

Raw images, patient-level records, local MLflow runs, ad hoc metrics, Grad-CAM outputs, and temporary inference outputs are intentionally ignored.

## Reproducibility Notes

The final experiment reports record the TensorFlow/Keras versions used when those reports were generated. `ml/requirements.txt` pins TensorFlow 2.19.1 as the supported public test/export environment because it is stable for the project on macOS arm64 and Linux Docker.

The experiment report references an earlier comparison run (`experiment_20260420T042800Z`) for context. That earlier run is not part of the public release artifact set; the public reproducibility baseline is the final Keras model plus the `20260420T043804Z` experiment and `20260420T045056Z` calibration reports.

Regenerate the TFLite artifact with:

```sh
cd ml
python scripts/export_tflite.py --overwrite
```
