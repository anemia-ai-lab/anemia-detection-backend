.PHONY: install run dev lint format test ml-test ml-venv ml-install ml-tf-check ml-test-docker \
	db-push ml-train-demo ml-train ml-eval ml-shell

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
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

test:
	env DISABLE_TF=1 INFERENCE_MODEL_PATH= $(TEST_PYTHON) -m pytest tests/

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

ml-eval:
	cd $(ML_DIR) && .venv/bin/python scripts/evaluate.py --model-path artifacts/models/baseline_mobilenetv2.keras --test-dir data/test

ml-shell:
	cd $(ML_DIR) && zsh