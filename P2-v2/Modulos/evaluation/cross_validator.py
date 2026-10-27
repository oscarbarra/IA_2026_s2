"""Búsqueda de hiperparámetros con validación cruzada estratificada.

Criterio de selección:
1. Maximizar la sensibilidad (recall) media de la clase positiva.
2. En caso de empate, maximizar la precisión media.

El ``refit`` se implementa como una función que recibe ``cv_results_`` y
devuelve el índice ganador, de modo que scikit-learn reentrena automáticamente
la mejor configuración sobre todo el conjunto de entrenamiento.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from Modulos.config import N_JOBS, N_SPLITS, POS_LABEL, SEED

# Métricas registradas en cv_results_. El nombre de la clave es el que usa
# scikit-learn: ``mean_test_recall``, ``mean_test_precision``, etc.
SCORING = {
    "recall": make_scorer(recall_score, pos_label=POS_LABEL, zero_division=0),
    "precision": make_scorer(
        precision_score, pos_label=POS_LABEL, zero_division=0
    ),
}


def _select_best_index(cv_results: dict) -> int:
    """Selecciona el índice con mayor recall y, a igualdad, mayor precisión."""
    recalls = np.nan_to_num(
        np.asarray(cv_results["mean_test_recall"], dtype=float), nan=-np.inf
    )
    precisions = np.nan_to_num(
        np.asarray(cv_results["mean_test_precision"], dtype=float), nan=-np.inf
    )
    # lexsort: la última clave es la primaria.
    return int(np.lexsort((precisions, recalls))[-1])


def run_search(
    estimator,
    param_distributions: dict,
    X,
    y,
    *,
    n_iter: int | None = None,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    n_jobs: int = N_JOBS,
    verbose: int = 1,
):
    """Ejecuta GridSearchCV o RandomizedSearchCV según ``n_iter``.

    Si ``n_iter`` es ``None`` se recorre la rejilla completa; en caso contrario
    se muestrean como máximo ``n_iter`` combinaciones (sin superar el tamaño
    total de la rejilla).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    common = dict(
        estimator=estimator,
        scoring=SCORING,
        cv=cv,
        refit=_select_best_index,
        n_jobs=n_jobs,
        verbose=verbose,
        error_score="raise",
        return_train_score=False,
    )

    if n_iter is not None:
        total_combinations = int(
            np.prod([len(values) for values in param_distributions.values()])
        )
        n_iter = min(int(n_iter), total_combinations)
        search = RandomizedSearchCV(
            param_distributions=param_distributions,
            n_iter=n_iter,
            random_state=seed,
            **common,
        )
    else:
        search = GridSearchCV(param_grid=param_distributions, **common)

    search.fit(X, y)
    return search


def summarize_search(search) -> dict:
    """Extrae los resultados relevantes de un objeto de búsqueda ya ajustado."""
    index = search.best_index_
    cv = search.cv_results_

    def value(key: str) -> float:
        return float(cv[key][index])

    return {
        "best_params": search.best_params_,
        "cv_recall_mean": value("mean_test_recall"),
        "cv_recall_std": value("std_test_recall"),
        "cv_precision_mean": value("mean_test_precision"),
        "cv_precision_std": value("std_test_precision"),
        "cv_fit_time_mean": value("mean_fit_time"),
        "cv_score_time_mean": value("mean_score_time"),
        "n_candidates": int(len(cv["params"])),
    }


def cv_results_dataframe(search, exp_id: str):
    """Devuelve un DataFrame resumido de ``cv_results_`` para exportación."""
    import pandas as pd

    columns = [
        "params",
        "mean_test_recall",
        "std_test_recall",
        "mean_test_precision",
        "std_test_precision",
        "rank_test_recall",
        "mean_fit_time",
        "mean_score_time",
    ]
    cv = search.cv_results_
    data = {column: cv[column] for column in columns}
    df = pd.DataFrame(data)
    df.insert(0, "exp_id", exp_id)
    return df.sort_values(
        ["mean_test_recall", "mean_test_precision"], ascending=False
    ).reset_index(drop=True)
