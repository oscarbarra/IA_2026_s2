"""Construcción de pipelines anti-data-leakage.

Se usa ``imblearn.pipeline.Pipeline`` porque, a diferencia del pipeline de
scikit-learn, permite remuestrear únicamente durante ``fit``. Así el
``RandomUnderSampler``/``SMOTE`` se ajusta en el fold de entrenamiento y no
transforma el fold de validación ni el conjunto de prueba.
"""

from __future__ import annotations

from imblearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from Modulos.config import DEFAULT_RESAMPLER, FEATURE_SCORE
from Modulos.feature_selector import build_feature_selector
from Modulos.imbalance_handler import build_resampler


def build_pipeline(
    estimator,
    *,
    scale: bool = True,
    feature_selection: bool = True,
    k_features: int | str = "all",
    feature_score: str = FEATURE_SCORE,
    resampler: str | None = DEFAULT_RESAMPLER,
    scaler: str = "standard",
) -> Pipeline:
    """Encadena escalado, selección, remuestreo y estimador.

    Orden (según el SDD):
        scaler -> feature selection -> resampler -> estimator

    Los pasos opcionales se omiten para que los árboles no reciban escalado,
    tal como indica la matriz experimental.
    """
    steps: list[tuple[str, object]] = []

    if scale:
        scaler_obj = RobustScaler() if scaler == "robust" else StandardScaler()
        steps.append(("scaler", scaler_obj))

    if feature_selection:
        steps.append(
            ("select", build_feature_selector(k_features, feature_score))
        )

    resampler_obj = build_resampler(resampler)
    if resampler_obj is not None:
        steps.append(("sampler", resampler_obj))

    steps.append(("model", estimator))
    return Pipeline(steps)
