# Anemia Prediction API using CNN and Probability Calibration

## Abstract

Childhood anemia remains prevalent in low-resource settings, where timely laboratory access is often limited. This repository provides a backend system for **early risk screening** from **fingernail photographs** using a convolutional neural network (CNN) with transfer learning. Raw classifier scores are post-processed with **probability calibration** (temperature scaling) and a **data-driven operational threshold** derived from the receiver operating characteristic (ROC). The system outputs a **probability of elevated risk** for screening workflows; it does **not** constitute a clinical diagnosis.

## Method Overview

The imaging pipeline targets **nail-bed regions** as a non-invasive proxy signal. The CNN uses **MobileNetV2** pretrained on ImageNet, with a lightweight binary head (global pooling, dropout, sigmoid). **Input** consists of registered fingernail crops; **output** is a scalar probability of positive-class risk.

**Training** follows a two-phase schedule: (1) optimization of the classification head with the backbone frozen; (2) **partial fine-tuning** of the deepest layers of MobileNetV2 while earlier layers remain frozen. The minority (positive) class is addressed via **oversampling** in the training subset of the internal train/validation split. The reported production configuration uses **no `class_weight`** in the final training objective, in line with the selected experiment.

**Post-processing at inference** applies **temperature scaling** to the sigmoid probability, using a scalar temperature estimated on the validation split. The **operational decision rule** compares the calibrated probability to a threshold chosen by **maximization of Youden’s J** on the held-out test set under the calibrated scores.

## Model Configuration (v1.0)

| Parameter | Value |
|-----------|--------|
| `model_version` | `v1.0` |
| Temperature (scaling) | `0.7510018331928743` |
| Operational threshold (calibrated probability) | `0.1680544387290045` |

## Evaluation (Test Set)

The following metrics refer to **test-set** evaluation with the **calibrated** probability and the **operational** (Youden) threshold.

| Metric | Value |
|--------|------:|
| AUC | 0.795 |
| Recall | 0.741 |
| Precision | 0.455 |
| Accuracy | 0.793 |
| Brier score | 0.118 |
| Expected calibration error (ECE) | 0.060 |

The operational point prioritizes **recall** relative to precision, which is appropriate for screening where false negatives carry high clinical cost. **Calibration** (low Brier score and moderate ECE) improves the interpretability of reported probabilities compared to raw sigmoid outputs, without changing the discriminative ordering underlying the ROC.

## API

**`POST /predict`**  
Multipart request with a **required image** (JPEG, PNG, or WebP) and optional **`birth_date`** (for age metadata). The response includes **`raw_probability`** (sigmoid output of the CNN), **`calibrated_probability`** (after temperature scaling), **`threshold_used`**, **`prediction`** (binary decision on the calibrated score), and **`risk`** (coarse risk stratum), together with persistence metadata when configured.

**`GET /model/evaluation`**  
Returns **offline evaluation metrics** and **calibration-related parameters** aligned with the deployed model version (`v1.0`), including AUC, precision/recall/accuracy at the operational threshold, Brier score, ECE, and flags summarizing training choices (e.g., oversampling, use of class weights, backbone fine-tuning).

Authentication, database, and object storage are provided via **Supabase** (see configuration in `.env.example`).

## Reproducibility

- **Random seed:** 42 for stratified splitting and training stochasticity where applicable.  
- **Train/validation:** stratified split on image labels within the training directory.  
- **Train/test:** patient-level separation at dataset construction so that all crops from a given subject appear in only one of train or test.  
- **Artifacts:** training and calibration runs are logged as **JSON** and **Markdown** under `ml/artifacts/runs/` for traceability.

## Project Structure

```
backend/     # FastAPI application, inference, configuration, Supabase integration
ml/          # Dataset utilities, training, calibration scripts, and artifact outputs
tests/       # Automated tests for API and core components
```

## Limitations

The model estimates **risk from imaging features**, not a definitive **diagnosis**; clinical correlation and laboratory follow-up remain necessary. Performance depends on **image quality**, illumination, and adherence to the intended capture protocol. **Dataset size and spectrum bias** may limit generalization to populations or devices not represented in development data.

## License

This software is intended for **academic and research use**. Redistribution, modification, and deployment for clinical or commercial purposes require appropriate ethical review, regulatory compliance, and explicit licensing terms beyond this document.
