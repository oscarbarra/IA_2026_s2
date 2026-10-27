"""Estrategias de compensación del desbalanceo de clases.

El remuestreo se aplica dentro del pipeline de ``imbalanced-learn``, de modo
que solo afecta a la partición de entrenamiento de cada fold. Nunca se aplica
al conjunto de validación ni al de prueba.
"""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from Modulos.config import SEED


def build_resampler(method: str | None = "under", seed: int = SEED):
    """Devuelve un remuestreador o ``None`` si no se desea compensar.

    Métodos disponibles:
    - ``"under"``: ``RandomUnderSampler`` (submuestreo de la clase mayoritaria).
    - ``"smote"``: ``SMOTE`` (sobremuestreo sintético de la clase minoritaria).
    - ``"none"`` / ``None``: sin compensación.
    """
    if method is None or str(method).lower() in {"none", "null"}:
        return None

    method = str(method).lower()
    if method == "under":
        return RandomUnderSampler(sampling_strategy=1.0, random_state=seed)
    if method == "smote":
        # k_neighbors=5 es seguro con el dataset completo; para submuestras
        # muy pequeñas puede ser necesario reducirlo.
        return SMOTE(sampling_strategy=1.0, k_neighbors=5, random_state=seed)

    raise ValueError(
        f"Método de compensación no soportado: '{method}'. "
        "Use 'under', 'smote' o 'none'."
    )
