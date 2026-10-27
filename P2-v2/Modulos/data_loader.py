"""Carga del dataset y auditoría inicial.

Responsabilidad única: leer el CSV, auditar su calidad y, opcionalmente,
extraer una submuestra estratificada para pruebas rápidas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from Modulos.config import DATA_PATH, MIN_SAMPLE_SIZE, POS_LABEL, SEED, TARGET_COLUMN
#from config import DATA_PATH, MIN_SAMPLE_SIZE, POS_LABEL, SEED, TARGET_COLUMN

def audit_data(df: pd.DataFrame) -> dict:
    """Devuelve un resumen de calidad y desbalance del dataset.

    Incluye dimensiones, duplicados, valores nulos, conteo por clase y la
    proporción de fraudes (clase positiva).
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET_COLUMN}' no existe.")

    counts = df[TARGET_COLUMN].value_counts().sort_index()
    frauds = int(counts.get(POS_LABEL, 0))
    total = int(len(df))

    return {
        "shape": df.shape,
        "duplicates": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "frauds": frauds,
        "fraud_ratio": (frauds / total) if total else 0.0,
        "imbalance_ratio": ((total - frauds) / frauds) if frauds else float("inf"),
        "n_features": int(df.shape[1] - 1),
    }


def print_audit(audit: dict) -> None:
    """Imprime la auditoría en un formato legible por consola."""
    print("=" * 70)
    print("AUDITORÍA DEL CONJUNTO DE DATOS")
    print("=" * 70)
    print(f"Dimensiones         : {audit['shape'][0]:,} filas x {audit['shape'][1]} columnas")
    print(f"Características     : {audit['n_features']}")
    print(f"Duplicados          : {audit['duplicates']:,}")
    print(f"Valores nulos       : {audit['missing_values']:,}")
    print(f"Conteo por clase    : {audit['class_counts']}")
    print(f"Fraudes (clase 1)   : {audit['frauds']:,} ({audit['fraud_ratio']:.4%})")
    print(f"Ratio de desbalanceo: 1 fraude por cada {audit['imbalance_ratio']:,.1f} legítimas")
    print("=" * 70)
    print()


def load_data(
    path: str | Path = DATA_PATH,
    sample_size: int | None = None,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Lee el CSV y aplica, si se indica, una submuestra estratificada.

    La submuestra conserva la proporción original de fraudes, por lo que es
    válida para pruebas rápidas sin romper la naturaleza desbalanceada.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en '{path}'. "
            "Descarga 'creditcard.csv' y colócalo en la carpeta Data/."
        )

    # Leer todo como float64 salvo la etiqueta, que se fuerza a entero.
    dtype = {TARGET_COLUMN: "int8"}
    df = pd.read_csv(path, dtype=dtype)

    if (sample_size is not None) and (sample_size < len(df)):
        if sample_size < MIN_SAMPLE_SIZE:
            raise ValueError(
                f"SAMPLE_SIZE debe ser al menos {MIN_SAMPLE_SIZE:,} registros "
                f"(recibido: {sample_size:,})."
            )
        df, _ = train_test_split( df, train_size=sample_size, random_state=seed, stratify=df[TARGET_COLUMN] )
        if verbose:
            print(f"[info] Usando submuestra estratificada de {len(df):,} registros.\n")

    if verbose:
        print_audit(audit_data(df))

    return df

if __name__ == "__main__":
    # Prueba rápida de carga y auditoría.
    load_data(verbose=True)