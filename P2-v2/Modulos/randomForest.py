"""Implementación propietaria de Random Forest.

Combina *Bagging* (muestras bootstrap con reemplazo) y *Random Subspace*
(subconjunto aleatorio de características en cada división) usando
``DecisionTreeClassifier`` como estimador base. No se utiliza
``RandomForestClassifier`` ni ``BaggingClassifier``.

La clase hereda de ``ClassifierMixin`` y ``BaseEstimator`` para integrarse
con ``Pipeline``, ``GridSearchCV`` y ``RandomizedSearchCV`` de scikit-learn.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted, validate_data


class CustomRandomForest(ClassifierMixin, BaseEstimator):
    """Random Forest con bootstrap y subespacio aleatorio.

    Parameters
    ----------
    n_estimators:
        Número de árboles del bosque.
    max_depth:
        Profundidad máxima de cada árbol (``None`` = sin límite).
    max_features:
        Número de características consideradas en cada división. Acepta los
        mismos valores que ``DecisionTreeClassifier`` (``"sqrt"``, ``"log2"``,
        un entero o una fracción).
    min_samples_split:
        Mínimo de muestras para dividir un nodo.
    min_samples_leaf:
        Mínimo de muestras en una hoja.
    random_state:
        Semilla global. A partir de ella se derivan semillas independientes
        para cada árbol, garantizando reproducibilidad.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: str | int | float | None = "sqrt",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------
    def fit(self, X, y) -> "CustomRandomForest":
        """Entrena un árbol por cada muestra bootstrap."""
        if self.n_estimators < 1:
            raise ValueError("n_estimators debe ser un entero mayor o igual a 1.")

        X, y = validate_data(self, X, y, dtype=np.float64)
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError(
                "CustomRandomForest necesita al menos dos clases para entrenar."
            )
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        # Semilla global -> una semilla independiente por árbol.
        rng = np.random.RandomState(self.random_state)
        tree_seeds = rng.randint(
            low=0, high=np.iinfo(np.int32).max, size=self.n_estimators
        )

        self.trees_: list[DecisionTreeClassifier] = []
        self.tree_features_: list[np.ndarray] = []

        for seed in tree_seeds:
            tree_rng = np.random.RandomState(int(seed))
            indices = self._bootstrap_indices(
                tree_rng, y, n_samples, self.classes_
            )
            X_boot, y_boot = X[indices], y[indices]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(seed),
            )
            tree.fit(X_boot, y_boot)

            self.trees_.append(tree)
            # El subespacio aleatorio se aplica en cada división mediante
            # ``max_features``; aquí se registran las características que el
            # árbol tuvo disponibles (todas las del problema).
            self.tree_features_.append(np.arange(self.n_features_in_))

        # Alias habitual en scikit-learn.
        self.estimators_ = self.trees_
        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.trees_], axis=0
        )
        return self

    @staticmethod
    def _bootstrap_indices(
        rng: np.random.RandomState,
        y: np.ndarray,
        n_samples: int,
        classes: np.ndarray,
        max_attempts: int = 10,
    ) -> np.ndarray:
        """Genera índices bootstrap con reemplazo.

        Si una muestra bootstrap no contiene todas las clases (posible en
        datasets muy pequeños), se reintenta. Como último recurso se fuerza la
        presencia de una observación de cada clase ausente.
        """
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        for _ in range(max_attempts):
            if np.unique(y[indices]).size == classes.size:
                return indices
            indices = rng.choice(n_samples, size=n_samples, replace=True)

        for cls in classes:
            if not np.any(y[indices] == cls):
                indices[rng.randint(n_samples)] = rng.choice(
                    np.flatnonzero(y == cls)
                )
        return indices

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------
    def _vote_counts(self, X: np.ndarray) -> np.ndarray:
        """Matriz (n_muestras, n_clases) con los votos de cada árbol."""
        check_is_fitted(self, "trees_")
        X = validate_data(self, X, reset=False)
        X = np.asarray(X)

        n_samples = X.shape[0]
        n_classes = self.classes_.size
        votes = np.zeros((n_samples, n_classes), dtype=np.int32)
        rows = np.arange(n_samples)

        for tree in self.trees_:
            predictions = tree.predict(X)
            class_idx = np.searchsorted(self.classes_, predictions)
            votes[rows, class_idx] += 1

        return votes

    def predict(self, X):
        """Votación mayoritaria. En caso de empate gana la clase positiva."""
        votes = self._vote_counts(X)
        max_votes = votes.max(axis=1)
        # Entre las clases empatadas se elige el índice mayor; con clases
        # [0, 1] esto prioriza la clase 1 (fraude), como pide el SDD.
        candidates = np.where(
            votes == max_votes[:, None],
            np.arange(self.classes_.size),
            -1,
        )
        return self.classes_[candidates.max(axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        """Probabilidad media de cada clase según los árboles."""
        check_is_fitted(self, "trees_")
        X = validate_data(self, X, reset=False)
        X = np.asarray(X)

        proba = np.zeros((X.shape[0], self.classes_.size), dtype=np.float64)
        for tree in self.trees_:
            tree_proba = tree.predict_proba(X)
            class_idx = np.searchsorted(self.classes_, tree.classes_)
            proba[:, class_idx] += tree_proba

        return proba / len(self.trees_)
