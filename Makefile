.PHONY: install run dev lint format test db-push ml-install ml-train-demo ml-train ml-eval ml-shell

PYTHON := python3
ML_DIR := ml
ML_PYTHON := .venv/bin/python

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
	$(PYTHON) -m pytest

db-push:
	supabase db push

ml-install:
	cd $(ML_DIR) && $(ML_PYTHON) -m pip install -r requirements.txt

ml-train-demo:
	cd $(ML_DIR) && $(ML_PYTHON) scripts/train.py --demo --head-epochs 1

ml-train:
	cd $(ML_DIR) && $(ML_PYTHON) scripts/train.py --train-dir data/train --fine-tune-epochs 2

ml-eval:
	cd $(ML_DIR) && $(ML_PYTHON) scripts/evaluate.py --model-path artifacts/models/baseline_mobilenetv2.keras --test-dir data/test

ml-shell:
	cd $(ML_DIR) && zsh