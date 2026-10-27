"""División estratificada entrenamiento/prueba.

El conjunto de prueba se reserva desde el inicio y no participa en ninguna
etapa de selección, escalado, remuestreo ni ajuste de hiperparámetros.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from Modulos.config import SEED, TARGET_COLUMN, TEST_SIZE


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    """Devuelve ``X_train, X_test, y_train, y_test`` de forma estratificada."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET_COLUMN}' no existe.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return train_test_split( X, y, test_size=test_size, random_state=seed, stratify=y )
