import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.naive_bayes import GaussianNB

def seleccion_manual_exploratoria(feature_names):
    # Basado en la exploración: El canal Rojo (R) distingue mejor madurez
    return [feature_names.index('R_mean')]

def seleccion_sfs(X_train, y_train):
    gnb = GaussianNB()
    sfs = SequentialFeatureSelector(gnb, n_features_to_select=1, direction='forward', cv=3)
    sfs.fit(X_train, y_train)
    return list(np.where(sfs.get_support())[0])