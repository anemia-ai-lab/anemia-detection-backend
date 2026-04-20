"""Rutas y hiperparámetros por defecto del baseline."""

from pathlib import Path

# Raíz del subproyecto `ml/` (padre del paquete `baseline`).
ML_ROOT: Path = Path(__file__).resolve().parent.parent

ARTIFACTS_ROOT: Path = ML_ROOT / "artifacts"
MODEL_DIR: Path = ARTIFACTS_ROOT / "models"
METRICS_DIR: Path = ARTIFACTS_ROOT / "metrics"
RUNS_DIR: Path = ARTIFACTS_ROOT / "runs"

DEFAULT_MODEL_NAME: str = "baseline_mobilenetv2.keras"
DEFAULT_METRICS_NAME: str = "latest_eval.json"

DATA_ROOT: Path = ML_ROOT / "data"
DEFAULT_TRAIN_DIR: Path = DATA_ROOT / "train"
DEFAULT_TEST_DIR: Path = DATA_ROOT / "test"

IMG_SIZE: tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32
SEED: int = 42

# Fase 1: cabezal sobre base congelada
HEAD_LEARNING_RATE: float = 1e-3
HEAD_EPOCHS: int = 5

# Fase 2: fine-tuning parcial (últimas capas del backbone)
FINE_TUNE_LEARNING_RATE: float = 1e-5
FINE_TUNE_EPOCHS: int = 0
FINE_TUNE_FREEZE_UP_TO_LAYER: int = -30  # últimas |n| capas entrenables
