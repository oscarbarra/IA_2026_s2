"""Matriz experimental de 8 configuraciones (4 clasificadores x 2 escenarios).

Cada experimento declara:
- si requiere escalado,
- si aplica compensación del desbalanceo,
- la fábrica del estimador,
- el espacio de hiperparámetros y el número de candidatos a muestrear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from Modulos import config
from Modulos.arbol import build_tree
from Modulos.knn import build_knn
from Modulos.pipeline_builder import build_pipeline
from Modulos.randomForest import CustomRandomForest
from Modulos.svm import build_svm


@dataclass(frozen=True)
class Experiment:
    """Definición declarativa de una configuración experimental."""

    exp_id: str
    classifier: str
    compensation: str
    balanced: bool
    scale: bool
    estimator_factory: Callable
    param_distributions: dict = field(default_factory=dict)
    n_iter: int | None = None
    resampler: str | None = None

    def build_pipeline(self, feature_score: str = config.FEATURE_SCORE):
        """Construye un pipeline nuevo (sin estado) para este experimento."""
        return build_pipeline(
            self.estimator_factory(),
            scale=self.scale,
            feature_selection=True,
            feature_score=feature_score,
            resampler=self.resampler,
        )


def _knn_params(quick: bool) -> tuple[dict, int]:
    if quick:
        return (
            {
                "model__n_neighbors": [3, 5, 7],
                "model__weights": ["uniform", "distance"],
                "select__k": [10, 20],
            },
            6,
        )
    return (
        {
            "model__n_neighbors": [3, 5, 7, 11],
            "model__weights": ["uniform", "distance"],
            "select__k": [10, 20, "all"],
        },
        10,
    )


def _tree_params(quick: bool) -> tuple[dict, int]:
    if quick:
        return (
            {
                "model__max_depth": [None, 10],
                "model__min_samples_leaf": [1, 5],
                "model__criterion": ["gini", "entropy"],
                "select__k": [10, 20],
            },
            6,
        )
    return (
        {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 5, 20],
            "model__criterion": ["gini", "entropy"],
            "select__k": [10, 20, "all"],
        },
        12,
    )


def _svm_params(quick: bool) -> tuple[dict, int]:
    if quick:
        return (
            {
                "model__C": [0.01, 0.1, 1.0],
                "select__k": [10, 20],
            },
            4,
        )
    return (
        {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "select__k": [10, 20, "all"],
        },
        8,
    )


def _rf_params(quick: bool) -> tuple[dict, int]:
    if quick:
        return (
            {
                "model__n_estimators": [20, 40],
                "model__max_depth": [None, 10],
                "model__max_features": ["sqrt", "log2"],
                "model__min_samples_leaf": [1, 5],
                "select__k": [10, 20],
            },
            4,
        )
    return (
        {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 10, 20],
            "model__max_features": ["sqrt", "log2"],
            "model__min_samples_leaf": [1, 5],
            "select__k": [10, 20, "all"],
        },
        8,
    )


def build_experiments( quick: bool = False, resampler: str = config.DEFAULT_RESAMPLER ) -> list[Experiment]:
    """Devuelve las 8 configuraciones de la matriz experimental."""
    knn_params, knn_iter = _knn_params(quick)
    tree_params, tree_iter = _tree_params(quick)
    svm_params, svm_iter = _svm_params(quick)
    rf_params, rf_iter = _rf_params(quick)

    balanced_label = f"Con compensación ({resampler})"

    def rf_factory() -> CustomRandomForest:
        return CustomRandomForest(random_state=config.SEED)

    experiments = [
        Experiment(
            exp_id="EXP_1",
            classifier="kNN",
            compensation="Sin compensación",
            balanced=False,
            scale=True,
            estimator_factory=build_knn,
            param_distributions=knn_params,
            n_iter=knn_iter,
            resampler=None,
        ),
        Experiment(
            exp_id="EXP_2",
            classifier="kNN",
            compensation=balanced_label,
            balanced=True,
            scale=True,
            estimator_factory=build_knn,
            param_distributions=knn_params,
            n_iter=knn_iter,
            resampler=resampler,
        ),
        Experiment(
            exp_id="EXP_3",
            classifier="Árbol de Decisión",
            compensation="Sin compensación",
            balanced=False,
            scale=False,
            estimator_factory=build_tree,
            param_distributions=tree_params,
            n_iter=tree_iter,
            resampler=None,
        ),
        Experiment(
            exp_id="EXP_4",
            classifier="Árbol de Decisión",
            compensation=balanced_label,
            balanced=True,
            scale=False,
            estimator_factory=build_tree,
            param_distributions=tree_params,
            n_iter=tree_iter,
            resampler=resampler,
        ),
        Experiment(
            exp_id="EXP_5",
            classifier="SVM",
            compensation="Sin compensación",
            balanced=False,
            scale=True,
            estimator_factory=build_svm,
            param_distributions=svm_params,
            n_iter=svm_iter,
            resampler=None,
        ),
        Experiment(
            exp_id="EXP_6",
            classifier="SVM",
            compensation=balanced_label,
            balanced=True,
            scale=True,
            estimator_factory=build_svm,
            param_distributions=svm_params,
            n_iter=svm_iter,
            resampler=resampler,
        ),
        Experiment(
            exp_id="EXP_7",
            classifier="Custom Random Forest",
            compensation="Sin compensación",
            balanced=False,
            scale=False,
            estimator_factory=rf_factory,
            param_distributions=rf_params,
            n_iter=rf_iter,
            resampler=None,
        ),
        Experiment(
            exp_id="EXP_8",
            classifier="Custom Random Forest",
            compensation=balanced_label,
            balanced=True,
            scale=False,
            estimator_factory=rf_factory,
            param_distributions=rf_params,
            n_iter=rf_iter,
            resampler=resampler,
        ),
    ]

    return experiments


# EXPERIMENT_IDS = [f"EXP_{i}" for i in range(1, 9)]
