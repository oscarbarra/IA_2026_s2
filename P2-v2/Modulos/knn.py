"""Fábrica del clasificador k-Nearest Neighbors."""

from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier

from Modulos.config import N_JOBS


def build_knn( *, n_neighbors: int = 5, weights: str = "distance", n_jobs: int = N_JOBS, ) -> KNeighborsClassifier:
    """Devuelve un k-NN sin entrenar (el pipeline se encarga del escalado)."""
    return KNeighborsClassifier( n_neighbors=n_neighbors, weights=weights, n_jobs=n_jobs )
