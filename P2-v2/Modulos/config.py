"""Constantes y parámetros globales del proyecto (principio DRY).

Centralizar aquí semillas, rutas y tamaños evita duplicar valores en los
módulos de datos, pipelines, validación y reporte.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
SEED: int = 42
RANDOM_STATE: int = SEED

# ---------------------------------------------------------------------------
# División de datos
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.30
TARGET_COLUMN: str = "Class"
POS_LABEL: int = 1

# ---------------------------------------------------------------------------
# Validación cruzada
# ---------------------------------------------------------------------------
N_SPLITS: int = 5

# ---------------------------------------------------------------------------
# Muestreo opcional para pruebas rápidas
# ---------------------------------------------------------------------------
# None => usar el dataset completo.
SAMPLE_SIZE: int | None = None
MIN_SAMPLE_SIZE: int = 10_000
QUICK_SAMPLE_SIZE: int = 15_000

# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
N_JOBS: int = -1
DEFAULT_RESAMPLER: str = "under"  # "under" | "smote" | "none"
FEATURE_SCORE: str = "f_classif"  # "mutual_info" | "f_classif"

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = ROOT_DIR / "Data" / "creditcard.csv"
OUTPUT_DIR: Path = ROOT_DIR / "Graficos"
