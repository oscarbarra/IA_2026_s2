
import numpy as np
from sklearn.decomposition import PCA

def aplicar_pca_y_evaluar(X_train, y_train, X_val, y_val, X_test, y_test):
    print("--- 3.6 Clasificación utilizando PCA ---")
    pca = PCA()
    pca.fit(X_train)
    
    var_exp = pca.explained_variance_ratio_
    var_cum = np.cumsum(var_exp)
    
    # Selección de componentes con varianza explicada > 90%
    n_components = np.argmax(var_cum >= 0.90) + 1
    print(f"Componentes seleccionadas (>=90% varianza): {n_components}")
    
    pca_opt = PCA(n_components=n_components)
    X_train_pca = pca_opt.fit_transform(X_train)
    X_val_pca = pca_opt.transform(X_val)
    X_test_pca = pca_opt.transform(X_test)
    
    return X_train_pca, X_val_pca, X_test_pca, n_components