"""Cálculo estandarizado de métricas de clasificación binaria.

Función pura: recibe etiquetas reales y predichas, y devuelve un diccionario
con todas las métricas solicitadas por el SDD, además de la matriz de
confusión desglosada.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_metrics(y_true, y_pred, pos_label: int = 1) -> dict:
    """Calcula sensibilidad, especificidad, precisión, F1, exactitud y FNR.

    La clase positiva es ``pos_label`` (por defecto, fraude = 1).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # La matriz siempre se devuelve con el orden estándar [[TN, FP], [FN, TP]]
    # para la clase 1, que es la que usan los reportes.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn_class1, fp_class1, fn_class1, tp_class1 = cm.ravel()

    if pos_label == 1:
        tn, fp, fn, tp = tn_class1, fp_class1, fn_class1, tp_class1
    else:
        tp, fn, fp, tn = tn_class1, fp_class1, fn_class1, tp_class1

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if cm.sum() else 0.0
    fnr = 1.0 - recall

    return {
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "fnr": float(fnr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
