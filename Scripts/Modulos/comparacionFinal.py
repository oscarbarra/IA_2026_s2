
import pandas as pd

def presentar_comparacion(res_manual, res_sfs, res_pca):
    print("\n=================== 3.7 COMPARACIÓN FINAL (SOBRE TEST) ===================")
    data = [
        {"Estrategia": "Bayes + Manual (Exploratoria)", **res_manual},
        {"Estrategia": "Bayes + SFS", **res_sfs},
        {"Estrategia": "PCA + Bayes", **res_pca}
    ]
    df_res = pd.DataFrame(data)
    cols = ['Estrategia', 'AUC', 'Accuracy', 'Sensibilidad', 'Especificidad', 'Threshold']
    df_clean = df_res[cols]
    print(df_clean.to_string(index=False))