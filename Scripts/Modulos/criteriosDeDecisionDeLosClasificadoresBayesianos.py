
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score

def entrenar_y_evaluar_roc(X_train, y_train, X_val, y_val, X_test, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    # Probabilidades de la clase positiva (Maduro = 1)
    probs_val = clf.predict_proba(X_val)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_val, probs_val)
    # Índice de Youden = TPR - FPR
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[best_idx]
    
    # Evaluación en Test con el umbral óptimo
    probs_test = clf.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= optimal_threshold).astype(int)
    
    fpr_t, tpr_t, _ = roc_curve(y_test, probs_test)
    roc_auc = auc(fpr_t, tpr_t)
    acc = accuracy_score(y_test, preds_test)
    sens = recall_score(y_test, preds_test, pos_label=1)
    spec = recall_score(y_test, preds_test, pos_label=0)
    
    metrics = {
        'AUC': roc_auc,
        'Accuracy': acc,
        'Sensibilidad': sens,
        'Especificidad': spec,
        'Threshold': optimal_threshold,
        'Model': clf
    }
    return metrics