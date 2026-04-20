from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
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


settings = Settings()
