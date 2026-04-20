install:
	python -m pip install -r requirements.txt

run:
	python -m uvicorn backend.main:app --reload

dev: install run

lint:
	python -m ruff check .

format:
	python -m ruff format .

test:
	python -m pytest

db-push:
	supabase db push