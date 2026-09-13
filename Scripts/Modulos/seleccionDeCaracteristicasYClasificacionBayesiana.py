
import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.naive_bayes import GaussianNB

def seleccion_manual_exploratoria(X, feature_names):
    # Basado en que H (Tono) y R (Rojo) diferencian mejor la madurez del tomate
    indices = [i for i, name in enumerate(feature_names) if name in ['R_mean', 'H_mean']]
    return indices

def seleccion_sfs(X_train, y_train):
    gnb = GaussianNB()
    sfs = SequentialFeatureSelector(gnb, n_features_to_select=2, direction='forward', cv=3)
    sfs.fit(X_train, y_train)
    indices = np.where(sfs.get_support())[0]
    return list(indices)