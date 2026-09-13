
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def calcular_jaccard(mask_pred, mask_true):
    intersection = np.logical_and(mask_pred, mask_true).sum()
    union = np.logical_or(mask_pred, mask_true).sum()
    return intersection / union if union != 0 else 0.0

def evaluar_kmeans_combinaciones(df_train, seed=42):
    """
    Evalúa R, G, B, RG, RB, GB, RGB utilizando K-Means (K=2) e índice Jaccard.
    """
    print("--- 3.2 Segmentación mediante K-Means ---")
    combinaciones = {
        'R': [0], 'G': [1], 'B': [2],
        'RG': [0, 1], 'RB': [0, 2], 'GB': [1, 2],
        'RGB': [0, 1, 2]
    }
    
    resultados = {k: [] for k in combinaciones.keys()}
    
    for _, row in df_train.iterrows():
        img = np.array(Image.open(row['img_path']).convert('RGB')) / 255.0
        mask_true = np.array(Image.open(row['mask_path']).convert('L')) > 128
        h, w, _ = img.shape
        
        for name, channels in combinaciones.items():
            X_pixels = img[:, :, channels].reshape(-1, len(channels))
            
            kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=seed)
            #labels = kmeans.fit_predict(X_pixels).reshape(h, w)
            idx_sample = np.random.choice(X_pixels.shape[0], size=5000, replace=False)
            kmeans.fit(X_pixels[idx_sample])
            labels = kmeans.predict(X_pixels).reshape(h, w)
            
            # Criterio de selección de cluster: El cluster con menor intensidad RGB promedio suele ser fondo
            c0_mean = img[labels == 0].mean() if np.any(labels == 0) else 0
            c1_mean = img[labels == 1].mean() if np.any(labels == 1) else 0
            fruit_cluster = 1 if c1_mean > c0_mean else 0
            
            mask_pred = (labels == fruit_cluster)
            jaccard = calcular_jaccard(mask_pred, mask_true)
            resultados[name].append(jaccard)
            
    promedios = {k: np.mean(v) for k, v in resultados.items()}
    mejor_comb = max(promedios, key=promedios.get)
    
    for k, v in promedios.items():
        print(f"Combinación {k}: Jaccard medio = {v:.4f}")
    print(f"Mejor combinación seleccionada: {mejor_comb}")
    
    return mejor_comb, combinaciones[mejor_comb]

def segmentar_imagen(img_np, channels, seed=42):
    h, w, _ = img_np.shape
    X_pixels = img_np[:, :, channels].reshape(-1, len(channels))
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=5, random_state=seed)
    labels = kmeans.fit_predict(X_pixels).reshape(h, w)
    
    c0_mean = img_np[labels == 0].mean() if np.any(labels == 0) else 0
    c1_mean = img_np[labels == 1].mean() if np.any(labels == 1) else 0
    fruit_cluster = 1 if c1_mean > c0_mean else 0
    return (labels == fruit_cluster)