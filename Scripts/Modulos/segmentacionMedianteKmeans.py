import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def segmentar_y_jaccard(img_np, mask_true, channels, seed=42):
    # Optimización: Reducción de resolución (100x100) solo para ajustar K-Means rápido
    img_low = np.array(Image.fromarray((img_np * 255).astype(np.uint8)).resize((100, 100))) / 255.0
    X_low = img_low[:, :, channels].reshape(-1, len(channels))
    
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1, max_iter=100, random_state=seed)
    kmeans.fit(X_low)
    
    # Inferencia en imagen original
    h, w, _ = img_np.shape
    X_full = img_np[:, :, channels].reshape(-1, len(channels))
    labels = kmeans.predict(X_full).reshape(h, w)
    
    # Criterio: El centroide con mayor suma de intensidad pertenece al fruto
    fruit_cluster = np.argmax(kmeans.cluster_centers_.sum(axis=1))
    mask_pred = (labels == fruit_cluster)
    
    intersection = np.logical_and(mask_pred, mask_true).sum()
    union = np.logical_or(mask_pred, mask_true).sum()
    jaccard = intersection / union if union != 0 else 0.0
    
    return mask_pred, jaccard

def evaluar_kmeans_combinaciones(df_train, seed=42):
    print("--- 3.2 Segmentación mediante K-Means ---")
    combinaciones = {
        'R': [0], 'G': [1], 'B': [2],
        'RG': [0, 1], 'RB': [0, 2], 'GB': [1, 2],
        'RGB': [0, 1, 2]
    }
    
    scores = {k: [] for k in combinaciones}
    
    for _, row in df_train.iterrows():
        img = np.array(Image.open(row['img_path']).convert('RGB')) / 255.0
        mask_true = np.array(Image.open(row['mask_path']).convert('L')) > 128
        
        for name, channels in combinaciones.items():
            _, jaccard = segmentar_y_jaccard(img, mask_true, channels, seed)
            scores[name].append(jaccard)
            
    promedios = {k: np.mean(v) for k, v in scores.items()}
    best_comb = max(promedios, key=promedios.get)
    
    for k, v in promedios.items():
        print(f"Combinación {k}: Jaccard = {v:.4f}")
    print(f"Mejor combinación: {best_comb}")
    
    return best_comb, combinaciones[best_comb]