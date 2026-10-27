"""Fábrica del clasificador Support Vector Machine.

Se utiliza ``LinearSVC`` en lugar de ``SVC`` con kernel RBF porque el dataset
tiene ~200.000 muestras de entrenamiento: un kernel RBF tendría coste
cuadrático y sería inviable en tiempo/memoria. ``LinearSVC`` es un SVM lineal
y escala de forma aproximadamente lineal con el número de muestras.
"""

from __future__ import annotations

from sklearn.svm import LinearSVC

from Modulos.config import SEED


def build_svm( *, C: float = 1.0, random_state: int = SEED, max_iter: int = 5_000 ) -> LinearSVC:
    """Devuelve un SVM lineal sin entrenar."""
    return LinearSVC( C=C, dual="auto", max_iter=max_iter, random_state=random_state )
