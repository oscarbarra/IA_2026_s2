import numpy as np
import pandas as pd
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

def verificar_valores_pca(X_train):
    print("\n================ VALORES PCA ================")
    
    # 1. Ajustar PCA con todos los componentes disponibles
    pca = PCA()
    pca.fit(X_train)
    
    # 2. Extraer varianza explicada y varianza acumulada
    var_explicada = pca.explained_variance_ratio_ * 100
    var_acumulada = np.cumsum(var_explicada)
    
    # 3. Crear DataFrame formateado para cotejar con la tabla
    df_pca = pd.DataFrame({
        'Componente': [f'PC{i+1}' for i in range(len(var_explicada))],
        'Var. Explicada (%)': np.round(var_explicada, 2),
        'Var. Acumulada (%)': np.round(var_acumulada, 2)
    })
    
    print(df_pca.to_string(index=False))
    
    # 4. Validar criterio >= 90%
    n_comp = np.argmax(var_acumulada >= 90.0) + 1
    print(f"\nComponentes retenidas para >= 90% de varianza: {n_comp}")
    print("======================================================================\n")

# Uso dentro de tu flujo principal:
# verificar_valores_pca(X_train)