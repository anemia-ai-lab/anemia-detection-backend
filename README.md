# Anemia Risk Prediction Backend

FastAPI backend and machine learning pipeline for early anemia risk prediction from childhood fingernail images.

Authored by Mateo Alejandro Vilchez Rios — Software Engineering, Backend and Machine Learning Systems.

## Abstract

This repository implements a research-oriented software system for non-invasive early anemia risk prediction using fingernail imagery. The system combines an authenticated API, calibrated model inference, offline TensorFlow Lite execution, and visual explainability support. Outputs are intended for risk screening and research workflows only; they are not medical diagnoses.

## Repository scope

Included here: FastAPI backend, Supabase integration (Auth, Postgres, Storage), runtime inference, ML training/export and preprocessing, published ML artifacts, SQL migrations, tests, OpenAPI, and architecture diagrams under `docs/architecture/`.

The mobile app is an external API client (HTTPS, multipart, JSON). This repo does **not** ship the mobile UI or store listings; field demos belong in the thesis or another repository.

Setup: [`docs/RUNBOOK.md`](docs/RUNBOOK.md). Responsibility ↔ path matrix: [`docs/TRACEABILITY.md`](docs/TRACEABILITY.md).

## System Overview

The system supports a mobile-oriented screening workflow in which an authenticated user photographs index, middle, and ring fingernails in a single frame. With connectivity, `POST /predict` runs MediaPipe-based nail detection, per-nail Keras ensemble inference, and max aggregation on the backend. The mobile app only captures and uploads the image.

The machine learning layer provides training, evaluation, model export, offline inference, and explainability tooling. Offline execution is supported through TensorFlow Lite artifacts and metadata so that mobile or field deployments can run predictions without continuous network access.

## Architecture

The repository is structured as a modular monolith. API routes remain thin, application logic is concentrated in services, persistence access is isolated in repositories, and external clients are wired through integration modules.

The backend uses Keras for server-side inference. Offline capability is provided by TensorFlow Lite artifacts plus metadata describing the required post-processing sequence. Probability calibration and thresholding are applied outside the neural network so that server and offline clients can preserve the same decision semantics.

## Key Capabilities

- Calibrated anemia risk prediction from fingernail images.
- Authenticated workflows backed by Supabase Auth, database, and storage.
- Offline inference through TensorFlow Lite for mobile or field settings.
- Grad-CAM visual explainability for research inspection of model attention patterns.
- Separate backend and ML validation paths to avoid TensorFlow dependency during API tests.
- Prometheus-compatible operational metrics for controlled deployments.

## Deployment

Production: **AWS ECS Fargate** (us-west-2) via **CDK Python** — Node 22 (`.nvmrc`), see [`docs/DEPLOYMENT_AWS.md`](docs/DEPLOYMENT_AWS.md).

## Documentation

| Document | Purpose |
|----------|---------|
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Local setup, tests, smoke, troubleshooting |
| [`docs/DEPLOYMENT_AWS.md`](docs/DEPLOYMENT_AWS.md) | Production ECS Fargate deployment |
| [`docs/RELEASE.md`](docs/RELEASE.md) | Release v1.0.0 and production smoke checklist |
| [`docs/TRACEABILITY.md`](docs/TRACEABILITY.md) | Thesis ↔ repository traceability matrix |
| [`docs/V1_VS_V2.md`](docs/V1_VS_V2.md) | Pediatric model v1 vs v2 comparison |
| [`ml/README.md`](ml/README.md) | ML pipeline, artifacts, and commands |
| [`ml/docs/MOBILE_INFERENCE.md`](ml/docs/MOBILE_INFERENCE.md) | Offline mobile inference contract |
| [`observability/README.md`](observability/README.md) | Local Prometheus/Grafana stack |
| [`infra/README.md`](infra/README.md) | CDK quickstart (delegates to DEPLOYMENT_AWS) |
| [`SECURITY.md`](SECURITY.md) | Vulnerability reporting policy |

## Technical Stack

- Python 3.11 recommended / used in CI (local `make run` uses `python3` from your PATH)
- FastAPI
- Pydantic
- Supabase
- TensorFlow / Keras
- TensorFlow Lite
- NumPy
- Pytest
- Ruff
- Docker

## Validation Strategy

Backend checks: contracts, auth, validation, prediction semantics, persistence errors, metrics, and behaviour when the model is unavailable. ML checks: TensorFlow/Keras, TFLite, preprocessing, Grad-CAM (see `ml/tests/`).

| Command | Purpose |
|---------|---------|
| `make test` | API suite with `DISABLE_TF=1` and empty `INFERENCE_MODEL_PATH` (no TF import for these tests). |
| `make lint` | Ruff. |
| `make ml-test` | `ml/tests/` — needs TensorFlow and `make ml-install`. |
| `make ml-test-docker` | Same ML tests in Linux Docker if the host TF install fails (e.g. some macOS setups); see [`docs/RUNBOOK.md`](docs/RUNBOOK.md). |

`DISABLE_TF` in `.env.example` documents the same flag outside `make test`. Tests prove software behaviour and artifact wiring, not clinical performance.

## Limitations

**Screening and research only:** outputs support **risk screening** and **research workflows** only.

**Not medical care:** this software does **not** provide **medical diagnosis**, **clinical confirmation**, **treatment recommendation**, or replacement for professional judgment. Clinical interpretation requires medical evaluation, laboratory confirmation, ethical oversight, and applicable regulatory review.

Prediction quality depends on image acquisition conditions, including focus, illumination, nail visibility, camera characteristics, and adherence to the intended capture protocol. Images outside the development distribution may reduce reliability.

Grad-CAM outputs are interpretability aids, not causal explanations. Heatmaps can be sensitive to architecture, selected layer, preprocessing, and gradient behavior, and should not be interpreted as clinical evidence.

Model performance may be affected by dataset size, population coverage, device variation, and spectrum bias. External validation is required before clinical, commercial, or public health use.

Offline inference requires strict alignment of preprocessing, calibration, thresholding, model version, and metadata with the validated server-side configuration.

## Reproducibility

Full setup, migrations, smoke checks, and ML pipeline commands: [`docs/RUNBOOK.md`](docs/RUNBOOK.md). Traceability matrix: [`docs/TRACEABILITY.md`](docs/TRACEABILITY.md). Release and production smoke: [`docs/RELEASE.md`](docs/RELEASE.md).

Pediatric model **v1 vs v2** (AUC, ensemble, risk tiers): [`docs/V1_VS_V2.md`](docs/V1_VS_V2.md).

The validation commands in the table above cover the main `make` targets. `requirements.txt` includes TensorFlow because production loads the bundled Keras ensemble; API tests set `DISABLE_TF=1` and clear model paths so they do not import TensorFlow.

## Public Release Notes

The repository includes the trained Keras model artifact and the exported TensorFlow Lite
artifact plus metadata for academic reproducibility of the research prototype. The clinical
image dataset, raw captures, patient-level records, and local environment secrets are not
included.

The included model artifact is provided for research inspection and software reproducibility only.
It does not authorize clinical use, diagnosis, or deployment without independent validation,
ethical review, and applicable regulatory assessment.

Supabase service-role credentials and metrics bearer tokens are server-side secrets. They are
represented only as empty variables in `.env.example` and must never be shipped to mobile or web
clients.

## Authorship

**Mateo Alejandro Vilchez Rios**  
Software Engineering — Backend and Machine Learning Systems  
Universidad Peruana de Ciencias Aplicadas (UPC)  
Contact: Available upon request  
Languages: English / Spanish

> AI systems must be designed to operate under uncertainty, not only under ideal conditions.

## License / Academic Usage Note

This software is released under the MIT License. The academic and medical-safety notes above limit
the intended interpretation of this research prototype; they do not establish clinical validity,
regulatory approval, or fitness for medical deployment.

Security issues: see [`SECURITY.md`](SECURITY.md).
