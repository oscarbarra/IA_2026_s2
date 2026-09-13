import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score
from sklearn.naive_bayes import GaussianNB

def entrenar_y_evaluar_roc(X_train, y_train, X_val, y_val, X_test, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    # Selección de umbral con Índice de Youden en Validación
    probs_val = clf.predict_proba(X_val)[:, 1]
    fpr_v, tpr_v, thresholds_v = roc_curve(y_val, probs_val)
    opt_thresh = thresholds_v[np.argmax(tpr_v - fpr_v)]
    
    # Evaluación final en Test
    probs_test = clf.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= opt_thresh).astype(int)
    
    fpr_t, tpr_t, _ = roc_curve(y_test, probs_test)
    
    return {
        'AUC': auc(fpr_t, tpr_t),
        'Accuracy': accuracy_score(y_test, preds_test),
        'Sensibilidad': recall_score(y_test, preds_test, pos_label=1),
        'Especificidad': recall_score(y_test, preds_test, pos_label=0),
        'Threshold': opt_thresh
    }