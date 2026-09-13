import pandas as pd

def presentar_comparacion(res_manual, res_sfs, res_pca):
    print("\n=================== 3.7 COMPARACIÓN FINAL (TEST) ===================")
    df_res = pd.DataFrame([
        {"Estrategia": "Bayes + Manual", **res_manual},
        {"Estrategia": "Bayes + SFS", **res_sfs},
        {"Estrategia": "PCA + Bayes", **res_pca}
    ])
    print(df_res[['Estrategia', 'AUC', 'Accuracy', 'Sensibilidad', 'Especificidad', 'Threshold']].to_string(index=False))