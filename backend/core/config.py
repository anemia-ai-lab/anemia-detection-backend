from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.schemas.model_evaluation import ModelEvalMetrics

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def repo_root() -> Path:
    """Raíz del repositorio (para rutas relativas a artefactos ML, etc.)."""
    return _REPO_ROOT


MODEL_VERSION = "v2.0"
RISK_THRESHOLD = 0.5

# Calibración alineada con el experimento de tesis (*temperature scaling* en validación).
INFERENCE_CALIBRATION_TEMPERATURE_DEFAULT = 1.405026093389256
INFERENCE_CALIBRATION_OPERATIONAL_THRESHOLD_DEFAULT = 0.3815443834698594
INFERENCE_RISK_TIER_LOW_UPPER_DEFAULT = 0.3243127259493805
INFERENCE_RISK_TIER_HIGH_LOWER_DEFAULT = 0.3815443834698594


def looks_like_placeholder(value: str) -> bool:
    v = value.strip()
    if not v:
        return True
    upper = v.upper()
    return upper == "REPLACE_ME" or "YOUR_" in upper or "CHANGE_ME" in upper


def supabase_url_is_valid(url: str) -> bool:
    u = url.strip()
    return u.startswith("https://") and ".supabase.co" in u


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
    metrics_bearer_token: str = Field(
        default="",
        validation_alias="METRICS_BEARER_TOKEN",
        description=(
            "Token opcional para proteger GET /metrics. En entornos no locales es obligatorio "
            "configurarlo para exponer métricas."
        ),
    )
    rate_limit_enabled: bool = Field(
        default=True,
        validation_alias="RATE_LIMIT_ENABLED",
        description="Activa límites en memoria para rutas costosas o sensibles.",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        validation_alias="RATE_LIMIT_WINDOW_SECONDS",
    )
    rate_limit_auth_requests: int = Field(
        default=20,
        ge=1,
        le=1000,
        validation_alias="RATE_LIMIT_AUTH_REQUESTS",
        description="Máximo de POST /auth/login y /auth/register por cliente y ventana.",
    )
    rate_limit_predict_requests: int = Field(
        default=30,
        ge=1,
        le=1000,
        validation_alias="RATE_LIMIT_PREDICT_REQUESTS",
        description="Máximo de POST /predict por cliente y ventana.",
    )

    trust_proxy_headers: bool = Field(
        default=False,
        validation_alias="TRUST_PROXY_HEADERS",
        description=(
            "Si es True, la IP para rate limiting puede tomarse de X-Forwarded-For (primer salto). "
            "Solo activar detrás de un proxy que sobrescriba o sanitice esa cabecera."
        ),
    )

    supabase_url: str = ""
    supabase_key: str = ""
    supabase_service_role_key: str = ""
    predictions_storage_bucket: str = Field(
        default="prediction-images",
        validation_alias="PREDICTIONS_STORAGE_BUCKET",
        description=(
            "Bucket de Storage para imágenes de predicción. Debe coincidir con el nombre "
            "creado en supabase/migrations (prediction-images). Cambiar solo con migración SQL."
        ),
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
        description=(
            "Umbral histórico (probabilidad sin calibrar). La decisión en ``POST /predict`` usa "
            "``inference_calibration_operational_threshold`` sobre la probabilidad **calibrada**."
        ),
    )

    inference_calibration_temperature: float = Field(
        default=INFERENCE_CALIBRATION_TEMPERATURE_DEFAULT,
        gt=0.0,
        validation_alias="INFERENCE_CALIBRATION_TEMPERATURE",
        description="Parámetro T de *temperature scaling* aplicado solo en inferencia (logit/T).",
    )

    inference_calibration_operational_threshold: float = Field(
        default=INFERENCE_CALIBRATION_OPERATIONAL_THRESHOLD_DEFAULT,
        ge=0.0,
        le=1.0,
        validation_alias="INFERENCE_CALIBRATION_OPERATIONAL_THRESHOLD",
        description=(
            "Umbral alto (τ_alto / high_lower): probabilidad calibrada ≥ umbral → riesgo ``high``. "
            "Coincide con Youden en validación cuando se sincroniza desde calibración."
        ),
    )

    inference_risk_tier_low_upper: float = Field(
        default=INFERENCE_RISK_TIER_LOW_UPPER_DEFAULT,
        ge=0.0,
        le=1.0,
        validation_alias="INFERENCE_RISK_TIER_LOW_UPPER",
        description="Umbral bajo (τ_bajo): probabilidad calibrada ≤ umbral → riesgo ``low``.",
    )

    inference_risk_tier_high_lower: float = Field(
        default=INFERENCE_RISK_TIER_HIGH_LOWER_DEFAULT,
        ge=0.0,
        le=1.0,
        validation_alias="INFERENCE_RISK_TIER_HIGH_LOWER",
        description="Umbral alto (τ_alto): probabilidad calibrada ≥ umbral → riesgo ``high``.",
    )

    inference_model_paths: str = Field(
        default="",
        validation_alias="INFERENCE_MODEL_PATHS",
        description=(
            "Rutas .keras separadas por coma para ensemble (promedio de raw_prob). "
            "Vacío: usar solo ``inference_model_path``."
        ),
    )

    inference_tta_enabled: bool = Field(
        default=False,
        validation_alias="INFERENCE_TTA_ENABLED",
        description="Si True, promedia raw_prob con flip horizontal (solo API).",
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
    prediction_image_max_pixels: int = Field(
        default=12_000_000,
        ge=64,
        le=100_000_000,
        validation_alias="PREDICTION_IMAGE_MAX_PIXELS",
        description="Límite duro de píxeles tras decodificar, antes de redimensionar.",
    )

    nail_presence_min_skin_ratio: float = Field(
        default=0.012,
        ge=0.0,
        le=0.5,
        validation_alias="NAIL_PRESENCE_MIN_SKIN_RATIO",
        description="Ratio mínimo de píxeles tipo piel (heurística previa a la CNN).",
    )

    inference_model_path: str = Field(
        default="ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed42.keras",
        validation_alias="INFERENCE_MODEL_PATH",
        description=(
            "Ruta al .keras entrenado (absoluta o relativa al repo). "
            "Vacío: no carga modelo; POST /predict devuelve 503 salvo predictor inyectado (tests)."
        ),
    )

    cors_allowed_origins: str = Field(
        default="",
        validation_alias="CORS_ALLOWED_ORIGINS",
        description=(
            "Orígenes CORS permitidos, separados por coma (p. ej. ``http://localhost:3000``). "
            "Vacío en ``development``: lista local reducida (Swagger / pruebas). "
            "Vacío en otros entornos: sin cabeceras CORS (adecuado para clientes nativos)."
        ),
    )

    model_eval: ModelEvalMetrics = Field(
        default_factory=ModelEvalMetrics,
        description="Métricas de evaluación (sin versión; la versión es model_version).",
    )

    @model_validator(mode="after")
    def _production_safety_checks(self) -> "Settings":
        env = self.environment.strip().lower()
        if env in ("production", "prod"):
            if self.debug:
                msg = "DEBUG=true is not allowed when APP_ENV=production"
                raise ValueError(msg)
            missing: list[str] = []
            if not self.supabase_url.strip():
                missing.append("SUPABASE_URL")
            if not self.supabase_key.strip():
                missing.append("SUPABASE_KEY")
            if not self.supabase_service_role_key.strip():
                missing.append("SUPABASE_SERVICE_ROLE_KEY")
            if not self.metrics_bearer_token.strip():
                missing.append("METRICS_BEARER_TOKEN")
            if missing:
                msg = "Production requires non-empty configuration: " + ", ".join(missing)
                raise ValueError(msg)
            invalid: list[str] = []
            if looks_like_placeholder(self.supabase_url) or not supabase_url_is_valid(self.supabase_url):
                invalid.append("SUPABASE_URL")
            if looks_like_placeholder(self.supabase_key):
                invalid.append("SUPABASE_KEY")
            if looks_like_placeholder(self.supabase_service_role_key):
                invalid.append("SUPABASE_SERVICE_ROLE_KEY")
            if looks_like_placeholder(self.metrics_bearer_token):
                invalid.append("METRICS_BEARER_TOKEN")
            if invalid:
                msg = (
                    "Production requires real values in Secrets Manager (not REPLACE_ME): "
                    + ", ".join(invalid)
                )
                raise ValueError(msg)
            bucket = self.predictions_storage_bucket.strip() or "prediction-images"
            if bucket != "prediction-images":
                msg = (
                    "PREDICTIONS_STORAGE_BUCKET must remain 'prediction-images' in production "
                    "unless Storage RLS migrations are updated."
                )
                raise ValueError(msg)
        return self

    def effective_cors_origins(self) -> list[str]:
        raw = self.cors_allowed_origins.strip()
        if raw:
            return [o.strip() for o in raw.split(",") if o.strip()]
        env = self.environment.strip().lower()
        if env in ("development", "dev", "local"):
            return [
                "http://127.0.0.1:3000",
                "http://localhost:3000",
                "http://127.0.0.1:5173",
                "http://localhost:5173",
                "http://127.0.0.1:8080",
                "http://localhost:8080",
                "http://127.0.0.1:8000",
                "http://localhost:8000",
            ]
        return []


settings = Settings()


def supabase_config_ready(
    *,
    supabase_url: str | None = None,
    supabase_key: str | None = None,
    supabase_service_role_key: str | None = None,
) -> bool:
    url = settings.supabase_url if supabase_url is None else supabase_url
    key = settings.supabase_key if supabase_key is None else supabase_key
    service_key = (
        settings.supabase_service_role_key
        if supabase_service_role_key is None
        else supabase_service_role_key
    )
    return (
        supabase_url_is_valid(url)
        and not looks_like_placeholder(key)
        and not looks_like_placeholder(service_key)
    )
