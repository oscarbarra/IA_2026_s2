"""Selección de características.

Encapsula ``SelectKBest`` para que pueda encadenarse dentro del pipeline sin
que el resto del código conozca los detalles de la función de puntuación.
"""

from __future__ import annotations

from functools import partial

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from Modulos.config import SEED


def build_feature_selector(
    k: int | str = "all",
    score_func: str = "mutual_info",
) -> SelectKBest:
    """Crea un selector de características.

    Parameters
    ----------
    k:
        Número de características a conservar o ``"all"`` para no descartar
        ninguna. El valor real puede sobrescribirse desde ``GridSearchCV``
        mediante la clave ``select__k``.
    score_func:
        ``"mutual_info"`` (información mutua, más costosa) o ``"f_classif"``
        (ANOVA F, mucho más rápida y adecuada para ejecuciones completas).
    """
    if score_func in {"mutual_info", "mutual_info_classif", "mi"}:
        scorer = partial(mutual_info_classif, random_state=SEED)
    elif score_func in {"f_classif", "anova"}:
        scorer = f_classif
    else:
        raise ValueError(
            f"score_func no soportado: '{score_func}'. "
            "Use 'mutual_info' o 'f_classif'."
        )

    return SelectKBest(score_func=scorer, k=k)
