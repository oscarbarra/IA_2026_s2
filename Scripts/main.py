import os
import glob
import time
import pandas as pd
from sklearn.model_selection import train_test_split

from Modulos.exploracionDeLosDatos import *
from Modulos.segmentacionMedianteKmeans import *
from Modulos.extraccionDeCaracterisiticasDelFruto import *
from Modulos.seleccionDeCaracteristicasYClasificacionBayesiana import *
from Modulos.criteriosDeDecisionDeLosClasificadoresBayesianos import *
from Modulos.clasificacionUtilizandoPCA import *
from Modulos.comparacionFinal import *

SEED = 42

def cargar_dataset():
    current_dir = os.getcwd()
    base_path = os.path.join(current_dir, "Scripts", "Dataset", "dataset")
    records = []
    
    for cat, label in [('ripe', 1), ('unripe', 0)]:
        img_dir = os.path.join(base_path, cat, "images")
        mask_dir = os.path.join(base_path, cat, "masks")
        for img_p in sorted(glob.glob(os.path.join(img_dir, "*.jpeg"))) + sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
            fname = os.path.splitext(os.path.basename(img_p))[0]
            mask_p = os.path.join(mask_dir, f"{fname}_mask.png")
            if os.path.exists(mask_p):
                records.append({'img_path': img_p, 'mask_path': mask_p, 'label': label})
                
    return pd.DataFrame(records)

def main():
    time_start = time.time()

    # 1. División de datos (60% Train, 20% Val, 20% Test)
    df = cargar_dataset()
    if df.empty:
        raise FileNotFoundError("Dataset no encontrado en 'Dataset/dataset/'.")
        
    df_train_val, df_test = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df['label'])
    df_train, df_val = train_test_split(df_train_val, test_size=0.25, random_state=SEED, stratify=df_train_val['label'])
    
    # 2. Exploración de Datos
    ejecutar_exploracion(df_train)
    
    # 3. Segmentación K-Means
    best_comb_name, best_channels = evaluar_kmeans_combinaciones(df_train, seed=SEED)
    
    # 4. Extracción de Características
    X_train, y_train, feat_names = extraer_caracteristicas_dataset(df_train, best_channels, seed=SEED)
    X_val, y_val, _ = extraer_caracteristicas_dataset(df_val, best_channels, seed=SEED)
    X_test, y_test, _ = extraer_caracteristicas_dataset(df_test, best_channels, seed=SEED)
    
    # 5. Selección de Características y Bayes
    idx_manual = seleccion_manual_exploratoria(feat_names)
    idx_sfs = seleccion_sfs(X_train, y_train)
    
    res_manual = entrenar_y_evaluar_roc(X_train[:, idx_manual], y_train, X_val[:, idx_manual], y_val, X_test[:, idx_manual], y_test)
    res_sfs = entrenar_y_evaluar_roc(X_train[:, idx_sfs], y_train, X_val[:, idx_sfs], y_val, X_test[:, idx_sfs], y_test)
    
    # 6. PCA + Bayes
    X_tr_pca, X_val_pca, X_test_pca, _ = aplicar_pca_y_evaluar(X_train, X_val, X_test)
    res_pca = entrenar_y_evaluar_roc(X_tr_pca, y_train, X_val_pca, y_val, X_test_pca, y_test)
    
    # 7. Comparación Final
    presentar_comparacion(res_manual, res_sfs, res_pca)

    time_end = time.time()
    print(f"\nTiempo total de ejecución: {time_end - time_start:.2f} segundos.")
if __name__ == "__main__":
    main()