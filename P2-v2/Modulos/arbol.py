"""Fábrica del clasificador Árbol de Decisión."""

from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier

from Modulos.config import SEED


def build_tree( *, random_state: int = SEED ) -> DecisionTreeClassifier:
    """Devuelve un árbol de decisión sin entrenar."""
    return DecisionTreeClassifier(random_state=random_state)
