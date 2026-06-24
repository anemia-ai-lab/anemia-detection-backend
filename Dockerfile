# Production-oriented image: FastAPI + TensorFlow CPU (ensemble Ghana v2 .keras).
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_VERSION=v2.0 \
    INFERENCE_MODEL_PATH=ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed42.keras \
    INFERENCE_MODEL_PATHS=ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed42.keras,ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed123.keras,ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed456.keras

# OpenMP for TensorFlow CPU wheels on Debian slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Layer cache: dependencies before application code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application; `repo_root()` in config.py is the working directory (/app)
COPY backend/ /app/backend/
COPY ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed42.keras \
    /app/ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed42.keras

# Non-root process
RUN useradd --create-home --uid 10001 app \
    && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
