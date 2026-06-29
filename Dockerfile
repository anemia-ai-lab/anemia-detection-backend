# Production-oriented image: FastAPI + TensorFlow CPU (ensemble Ghana v2 .keras).
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10 \
    PYTHONPATH=/app \
    MODEL_VERSION=v2.0 \
    INFERENCE_MODEL_PATH=ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed42.keras \
    INFERENCE_MODEL_PATHS=ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed42.keras,ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed123.keras,ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed456.keras

# OpenMP for TensorFlow CPU wheels on Debian slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libgles2-mesa \
        libegl1-mesa \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Layer cache: dependencies before application code (TensorFlow ~645 MB in its own layer)
COPY requirements.txt .
RUN pip install --default-timeout=1000 --retries=10 --no-cache-dir tensorflow==2.19.1
RUN pip install --default-timeout=300 --retries=10 --no-cache-dir opencv-python-headless==4.11.0.86
RUN pip install --default-timeout=300 --retries=10 --no-cache-dir -r requirements.txt
RUN pip install --default-timeout=300 --retries=10 --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.11.0.86

# Application; `repo_root()` in config.py is the working directory (/app)
COPY backend/ /app/backend/
COPY ml/__init__.py /app/ml/__init__.py
COPY ml/preprocessing/ /app/ml/preprocessing/
COPY ml/baseline/ /app/ml/baseline/
RUN mkdir -p /app/ml/artifacts/models \
    && curl -fsSL \
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" \
        -o /app/ml/artifacts/models/hand_landmarker.task
COPY ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed42.keras \
    ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed123.keras \
    ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed456.keras \
    /app/ml/artifacts/models/

# Non-root process
RUN useradd --create-home --uid 10001 app \
    && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
