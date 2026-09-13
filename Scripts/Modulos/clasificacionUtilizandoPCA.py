import numpy as np
from sklearn.decomposition import PCA

def aplicar_pca_y_evaluar(X_train, X_val, X_test):
    print("--- 3.6 Clasificación utilizando PCA ---")
    pca = PCA()
    pca.fit(X_train)
    
    # Retener componentes para explicar >= 90% de varianza
    var_cum = np.cumsum(pca.explained_variance_ratio_)
    n_comp = np.argmax(var_cum >= 0.90) + 1
    
    pca_opt = PCA(n_components=n_comp)
    X_tr_pca = pca_opt.fit_transform(X_train)
    X_val_pca = pca_opt.transform(X_val)
    X_test_pca = pca_opt.transform(X_test)
    
    return X_tr_pca, X_val_pca, X_test_pca, n_comp