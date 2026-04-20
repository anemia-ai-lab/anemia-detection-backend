from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.schemas.model_evaluation import ModelEvalMetrics

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def repo_root() -> Path:
    """Raíz del repositorio (para rutas relativas a artefactos ML, etc.)."""
    return _REPO_ROOT


MODEL_VERSION = "v1.0"
RISK_THRESHOLD = 0.5


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    app_name: str = "Anemia Detection API"
    environment: str = Field(default="development", validation_alias="APP_ENV")
    debug: bool = False

    supabase_url: str = ""
    supabase_key: str = ""
    supabase_service_role_key: str = ""
    predictions_storage_bucket: str = Field(
        default="prediction-images",
        validation_alias="PREDICTIONS_STORAGE_BUCKET",
    )

    model_version: str = Field(
        default=MODEL_VERSION,
        validation_alias="MODEL_VERSION",
        description="Versión del modelo usada en predicciones y en GET /model/evaluation.",
    )

    risk_threshold: float = Field(
        default=RISK_THRESHOLD,
        ge=0.0,
        le=1.0,
        validation_alias="RISK_THRESHOLD",
        description="Umbral de probabilidad: score >= umbral → high, si no → low.",
    )

    prediction_image_max_bytes: int = Field(
        default=5 * 1024 * 1024,
        ge=1024,
        le=50 * 1024 * 1024,
        validation_alias="PREDICTION_IMAGE_MAX_BYTES",
        description="Tamaño máximo del fichero de imagen para POST /predict (bytes).",
    )

    prediction_image_max_edge_px: int = Field(
        default=1024,
        ge=256,
        le=4096,
        validation_alias="PREDICTION_IMAGE_MAX_EDGE_PX",
        description="Lado máximo en píxeles tras decodificar (antes de uña y CNN).",
    )

    nail_presence_min_skin_ratio: float = Field(
        default=0.012,
        ge=0.0,
        le=0.5,
        validation_alias="NAIL_PRESENCE_MIN_SKIN_RATIO",
        description="Ratio mínimo de píxeles tipo piel (heurística previa a la CNN).",
    )

    inference_model_path: str = Field(
        default="ml/artifacts/models/baseline_mobilenetv2.keras",
        validation_alias="INFERENCE_MODEL_PATH",
        description=(
            "Ruta al .keras entrenado (absoluta o relativa al repo). "
            "Vacío: no carga modelo; POST /predict devuelve 503 salvo predictor inyectado (tests)."
        ),
    )

    model_eval: ModelEvalMetrics = Field(
        default_factory=ModelEvalMetrics,
        description="Métricas de evaluación (sin versión; la versión es model_version).",
    )


settings = Settings()
