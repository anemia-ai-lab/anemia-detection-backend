.PHONY: install run dev lint format test smoke-prod ml-test ml-venv ml-install ml-tf-check ml-test-docker \
	db-push ml-train-demo ml-train ml-train-finetune ml-eval ml-eval-ghana \
	ml-prepare-ghana ml-docker-eval-ghana ml-docker-train-finetune ml-docker-finetune-ghana \
	ml-docker-train-ghana-scratch ml-docker-calibrate-ghana \
	ml-docker-prepare-ghana-augmented ml-docker-train-ghana-augmented \
	ml-docker-train-ghana-augmented-seed ml-docker-train-ghana-ensemble-seeds \
	ml-docker-calibrate-ghana-augmented-tiers ml-docker-calibrate-ensemble-ghana \
	ml-docker-export-ensemble-tflite 	ml-docker-train-ghana-focal ml-docker-train-ghana-mobile-aug \
	ml-docker-eval-ghana-original-only-v1 ml-docker-eval-ghana-original-only-v2 \
	ml-shell

ML_DOCKER_RUN = docker run --rm -v "$(PWD):/workspace" -w /workspace/ml -e PYTHONPATH=/workspace $(ML_TEST_IMAGE)
GHANA_AUG_BASELINE_JSON = artifacts/runs/experiment_20260601T050706Z.json
GHANA_ENSEMBLE_SEEDS = 42 123 456

PYTHON := python3
TEST_PYTHON := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo $(PYTHON))
ML_DIR := ml
ML_PYTHON := $(ML_DIR)/.venv/bin/python
ML_TEST_IMAGE ?= anemia-ml-test

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m uvicorn backend.main:app --reload

dev: install run

lint:
	$(TEST_PYTHON) -m ruff check .
	
format:
	$(TEST_PYTHON) -m ruff format .

test:
	env DISABLE_TF=1 INFERENCE_MODEL_PATH= $(TEST_PYTHON) -m pytest tests/

smoke-prod:
	$(PYTHON) scripts/smoke_prod.py

ml-test:
	PYTHONPATH=. $(ML_PYTHON) -m pytest ml/tests/

ml-venv:
	@test -x $(ML_PYTHON) || ( \
		cd $(ML_DIR) && ( \
			(command -v python3.11 >/dev/null 2>&1 && python3.11 -m venv .venv) || \
			(command -v python3 >/dev/null 2>&1 && python3 -m venv .venv) \
		) \
	)

ml-install: ml-venv
	$(ML_PYTHON) -m pip install -U pip setuptools wheel
	$(ML_PYTHON) -m pip install -r $(ML_DIR)/requirements.txt

ml-tf-check:
	@PYTHONPATH=. $(ML_PYTHON) -c "import tensorflow as tf; print(tf.__version__)"

ml-test-docker:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	docker run --rm $(ML_TEST_IMAGE)

db-push:
	supabase db push

ml-train-demo:
	cd $(ML_DIR) && .venv/bin/python scripts/train.py --demo --head-epochs 1

ml-train:
	cd $(ML_DIR) && .venv/bin/python scripts/train.py --train-dir data/train --fine-tune-epochs 2

ml-train-finetune:
	cd $(ML_DIR) && .venv/bin/python scripts/train.py \
		--train-dir data/train --test-dir data/test \
		--metadata-path data_raw/nature/metadata.csv \
		--fine-tune-epochs 10

ml-prepare-ghana:
	cd $(ML_DIR) && .venv/bin/python scripts/prepare_ghana_dataset.py

ml-eval:
	cd $(ML_DIR) && .venv/bin/python scripts/evaluate.py \
		--model-path artifacts/models/baseline_mobilenetv2_ghana_augmented_seed42.keras \
		--test-dir data/ghana/test

ml-eval-ghana:
	cd $(ML_DIR) && .venv/bin/python scripts/evaluate_dir.py \
		--test-dir data/ghana/test \
		--calibration-json artifacts/runs/calibration_ensemble_ghana_v2.json \
		--dataset-label ghana_ensemble_v2

# TensorFlow en Docker (monta repo: incluye ml/data y ml/data_raw gitignored en imagen).
ml-docker-eval-ghana:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	docker run --rm -v "$(PWD):/workspace" -w /workspace/ml -e PYTHONPATH=/workspace \
		$(ML_TEST_IMAGE) python scripts/evaluate_dir.py \
		--test-dir data/ghana/test \
		--calibration-json artifacts/runs/calibration_ensemble_ghana_v2.json \
		--dataset-label ghana_ensemble_v2

ml-docker-train-finetune:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	docker run --rm -v "$(PWD):/workspace" -w /workspace/ml -e PYTHONPATH=/workspace \
		$(ML_TEST_IMAGE) python scripts/train.py \
		--train-dir data/train --test-dir data/test \
		--metadata-path data_raw/nature/metadata.csv \
		--fine-tune-epochs 10

# Ablation legacy: transfer Nature→Ghana (requiere modelo Nature local, no versionado en git).
ml-docker-finetune-ghana:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	docker run --rm -v "$(PWD):/workspace" -w /workspace/ml -e PYTHONPATH=/workspace \
		$(ML_TEST_IMAGE) python scripts/train.py \
		--init-model artifacts/models/baseline_mobilenetv2.keras \
		--train-dir data/ghana/train --test-dir data/ghana/test \
		--head-epochs 0 --fine-tune-epochs 10 \
		--output-model artifacts/models/baseline_mobilenetv2.keras

ml-docker-train-ghana-scratch:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	docker run --rm -v "$(PWD):/workspace" -w /workspace/ml -e PYTHONPATH=/workspace \
		$(ML_TEST_IMAGE) python scripts/train.py \
		--train-dir data/ghana/train --test-dir data/ghana/test \
		--head-epochs 5 --fine-tune-epochs 10 \
		--experiment-tag ghana_scratch \
		--baseline-experiment-json artifacts/runs/experiment_20260601T045054Z.json \
		--output-model artifacts/models/baseline_mobilenetv2_ghana.keras

ml-docker-calibrate-ghana:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	docker run --rm -v "$(PWD):/workspace" -w /workspace/ml -e PYTHONPATH=/workspace \
		$(ML_TEST_IMAGE) python scripts/calibrate_eval.py \
		--model-path artifacts/models/baseline_mobilenetv2_ghana.keras \
		--train-dir data/ghana/train --test-dir data/ghana/test

ml-docker-prepare-ghana-augmented:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	docker run --rm -v "$(PWD):/workspace" -w /workspace/ml -e PYTHONPATH=/workspace \
		$(ML_TEST_IMAGE) python scripts/prepare_ghana_dataset.py --include-augmented

ml-docker-train-ghana-augmented:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	docker run --rm -v "$(PWD):/workspace" -w /workspace/ml -e PYTHONPATH=/workspace \
		$(ML_TEST_IMAGE) python scripts/train.py \
		--train-dir data/ghana/train --test-dir data/ghana/test \
		--head-epochs 5 --fine-tune-epochs 10 \
		--experiment-tag ghana_scratch_augmented \
		--baseline-experiment-json artifacts/runs/experiment_20260601T050706Z.json \
		--output-model artifacts/models/baseline_mobilenetv2_ghana_augmented.keras

# Ensemble: una semilla por invocación, p. ej. make ml-docker-train-ghana-augmented-seed SEED=123
SEED ?= 42
ml-docker-train-ghana-augmented-seed:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	$(ML_DOCKER_RUN) python scripts/train.py \
		--train-dir data/ghana/train --test-dir data/ghana/test \
		--head-epochs 5 --fine-tune-epochs 10 --seed $(SEED) \
		--experiment-tag ghana_scratch_augmented_seed$(SEED) \
		--baseline-experiment-json $(GHANA_AUG_BASELINE_JSON) \
		--output-model artifacts/models/baseline_mobilenetv2_ghana_augmented_seed$(SEED).keras

ml-docker-train-ghana-ensemble-seeds:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	@for s in $(GHANA_ENSEMBLE_SEEDS); do \
		echo ">> Entrenando Ghana augmented seed=$$s"; \
		$(ML_DOCKER_RUN) python scripts/train.py \
			--train-dir data/ghana/train --test-dir data/ghana/test \
			--head-epochs 5 --fine-tune-epochs 10 --seed $$s \
			--experiment-tag ghana_scratch_augmented_seed$$s \
			--baseline-experiment-json $(GHANA_AUG_BASELINE_JSON) \
			--output-model artifacts/models/baseline_mobilenetv2_ghana_augmented_seed$$s.keras || exit 1; \
	done

ml-docker-calibrate-ghana-augmented-tiers:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	$(ML_DOCKER_RUN) python scripts/calibrate_eval.py \
		--model-path artifacts/models/baseline_mobilenetv2_ghana_augmented.keras \
		--train-dir data/ghana/train --test-dir data/ghana/test

ml-docker-calibrate-ensemble-ghana:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	$(ML_DOCKER_RUN) python scripts/calibrate_ensemble_eval.py \
		--model-path artifacts/models/baseline_mobilenetv2_ghana_augmented_seed42.keras \
		--model-path artifacts/models/baseline_mobilenetv2_ghana_augmented_seed123.keras \
		--model-path artifacts/models/baseline_mobilenetv2_ghana_augmented_seed456.keras \
		--train-dir data/ghana/train --test-dir data/ghana/test

ml-docker-export-ensemble-tflite:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	$(ML_DOCKER_RUN) python scripts/export_ensemble_tflite.py \
		--keras-path artifacts/models/baseline_mobilenetv2_ghana_augmented_seed42.keras \
		--keras-path artifacts/models/baseline_mobilenetv2_ghana_augmented_seed123.keras \
		--keras-path artifacts/models/baseline_mobilenetv2_ghana_augmented_seed456.keras \
		--calibration-json artifacts/runs/calibration_ensemble_ghana_v2.json \
		--overwrite

ml-docker-train-ghana-focal:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	$(ML_DOCKER_RUN) python scripts/train.py \
		--train-dir data/ghana/train --test-dir data/ghana/test \
		--head-epochs 5 --fine-tune-epochs 10 --seed 42 --focal-loss \
		--experiment-tag ghana_scratch_augmented_focal \
		--baseline-experiment-json $(GHANA_AUG_BASELINE_JSON) \
		--output-model artifacts/models/baseline_mobilenetv2_ghana_augmented_focal.keras

ml-docker-train-ghana-mobile-aug:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	$(ML_DOCKER_RUN) python scripts/train.py \
		--train-dir data/ghana/train --test-dir data/ghana/test \
		--head-epochs 5 --fine-tune-epochs 10 --seed 42 --mobile-capture-augment \
		--experiment-tag ghana_scratch_augmented_mobile_capture \
		--baseline-experiment-json $(GHANA_AUG_BASELINE_JSON) \
		--output-model artifacts/models/baseline_mobilenetv2_ghana_augmented_mobile_capture.keras

ml-docker-eval-ghana-original-only-v1:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	$(ML_DOCKER_RUN) python scripts/evaluate_ghana_original_only.py \
		--model-path artifacts/models/baseline_mobilenetv2_ghana_augmented.keras \
		--calibration-json artifacts/runs/calibration_20260601T052254Z.json

ml-docker-eval-ghana-original-only-v2:
	docker build -f Dockerfile.ml-test -t $(ML_TEST_IMAGE) .
	$(ML_DOCKER_RUN) python scripts/evaluate_ghana_original_only.py \
		--ensemble \
		--model-path artifacts/models/baseline_mobilenetv2_ghana_augmented_seed42.keras \
		--model-path artifacts/models/baseline_mobilenetv2_ghana_augmented_seed123.keras \
		--model-path artifacts/models/baseline_mobilenetv2_ghana_augmented_seed456.keras \
		--calibration-json artifacts/runs/calibration_ensemble_ghana_v2.json

ml-shell:
	cd $(ML_DIR) && zsh

.PHONY: c4-code-generate c4-code-render c4-code-clean

C4_CODE_DIR := docs/architecture/code
C4_CODE_GENERATED := $(C4_CODE_DIR)/_generated
C4_CODE_RENDERED := $(C4_CODE_DIR)/rendered
C4_CODE_PACKAGES := services repositories inference core api schemas

c4-code-generate:
	mkdir -p $(C4_CODE_GENERATED)
	docker run --rm -v "$(PWD):/app" -w /app python:3.11-slim sh -c ' \
		python -m pip install --quiet pylint && \
		for pkg in $(C4_CODE_PACKAGES); do \
			echo ">> pyreverse backend.$$pkg"; \
			python -m pylint.pyreverse.main \
				-o puml -p $$pkg \
				-d $(C4_CODE_GENERATED) \
				backend.$$pkg || true; \
		done'

c4-code-render:
	mkdir -p $(C4_CODE_RENDERED)
	docker run --rm -v "$(PWD)/$(C4_CODE_DIR):/data" plantuml/plantuml \
		-tsvg -o rendered "/data/*.puml"

c4-code-clean:
	rm -rf $(C4_CODE_GENERATED) $(C4_CODE_RENDERED)

.PHONY: c4-structurizr

c3-structurizr:
	docker run -it --rm \
		-p 8080:8080 \
		-v "$(PWD)/docs/architecture:/usr/local/structurizr" \
		structurizr/structurizr local