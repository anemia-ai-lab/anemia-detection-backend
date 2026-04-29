# Anemia Risk Prediction Backend

FastAPI backend and machine learning pipeline for early anemia risk prediction from childhood fingernail images.

Authored by Mateo Alejandro Vilchez Rios — Software Engineering, Backend and Machine Learning Systems.

## Abstract

This repository implements a research-oriented software system for non-invasive early anemia risk prediction using fingernail imagery. The system combines an authenticated API, calibrated model inference, offline TensorFlow Lite execution, and visual explainability support. Outputs are intended for risk screening and research workflows only; they are not medical diagnoses.

## System Overview

The system supports a mobile-oriented screening workflow in which an authenticated user submits a fingernail image and receives a calibrated risk prediction. The backend validates input images, executes model inference when a Keras artifact is configured, applies probability calibration and an operational decision threshold, and persists prediction metadata through Supabase.

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

## Technical Stack

- Python 3.11
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

Backend validation focuses on API contracts, authentication boundaries, request validation, prediction response semantics, persistence error handling, operational metrics, and model-unavailable behavior. The backend suite is designed to run without importing TensorFlow, keeping API tests fast and stable.

Machine learning validation is separated from backend validation. ML tests exercise TensorFlow/Keras behavior, TensorFlow Lite compatibility, preprocessing consistency, and Grad-CAM behavior. Docker-based execution is available for environments where host TensorFlow wheels are unstable.

The validation strategy distinguishes software correctness from clinical validity. Passing tests confirms implementation behavior and artifact compatibility; it does not establish diagnostic performance in clinical deployment.

## Limitations

This system predicts risk and must not be used as a medical diagnosis. Clinical interpretation requires medical evaluation, laboratory confirmation, ethical oversight, and applicable regulatory review.

Prediction quality depends on image acquisition conditions, including focus, illumination, nail visibility, camera characteristics, and adherence to the intended capture protocol. Images outside the development distribution may reduce reliability.

Grad-CAM outputs are interpretability aids, not causal explanations. Heatmaps can be sensitive to architecture, selected layer, preprocessing, and gradient behavior, and should not be interpreted as clinical evidence.

Model performance may be affected by dataset size, population coverage, device variation, and spectrum bias. External validation is required before clinical, commercial, or public health use.

Offline inference requires strict alignment of preprocessing, calibration, thresholding, model version, and metadata with the validated server-side configuration.

## Reproducibility

Run backend tests without TensorFlow:

```bash
make test
```

Run linting:

```bash
make lint
```

Prepare the ML environment:

```bash
make ml-install
```

Check TensorFlow availability in the ML environment:

```bash
make ml-tf-check
```

Run ML tests locally:

```bash
make ml-test
```

Run ML tests in Docker:

```bash
make ml-test-docker
```

## Public Release Notes

The repository includes the trained Keras model artifact for academic reproducibility of the
research prototype. The clinical image dataset, raw captures, patient-level records, and local
environment secrets are not included.

The included model artifact is provided for research inspection and software reproducibility only.
It does not authorize clinical use, diagnosis, or deployment without independent validation,
ethical review, and applicable regulatory assessment.

## Authorship

**Mateo Alejandro Vilchez Rios**  
Software Engineering — Backend and Machine Learning Systems  
Universidad Peruana de Ciencias Aplicadas (UPC)  
Contact: Available upon request  
Languages: English / Spanish

> AI systems must be designed to operate under uncertainty, not only under ideal conditions.

## License / Academic Usage Note

This software is intended for academic and research use. Any clinical, commercial, or public health deployment requires independent validation, ethical review, regulatory assessment, data protection review, and explicit licensing terms beyond this repository.
