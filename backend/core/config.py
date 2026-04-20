from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.schemas.model_evaluation import ModelEvalMetrics

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

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

    model_eval: ModelEvalMetrics = Field(
        default_factory=ModelEvalMetrics,
        description="Métricas de evaluación (sin versión; la versión es model_version).",
    )


settings = Settings()
