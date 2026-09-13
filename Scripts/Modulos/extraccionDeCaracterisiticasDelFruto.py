import numpy as np
from PIL import Image
from Modulos.segmentacionMedianteKmeans import segmentar_y_jaccard

def extraer_caracteristicas_dataset(df, best_channels, seed=42):
    print("--- 3.3 Extracción de Características ---")
    features, labels = [], []
    
    for _, row in df.iterrows():
        img = np.array(Image.open(row['img_path']).convert('RGB')) / 255.0
        mask_true = np.array(Image.open(row['mask_path']).convert('L')) > 128
        
        mask_pred, _ = segmentar_y_jaccard(img, mask_true, best_channels, seed)
        
        if not np.any(mask_pred):
            mask_pred = np.ones((img.shape[0], img.shape[1]), dtype=bool)
            
        mean_rgb = img[mask_pred].mean(axis=0)
        features.append(mean_rgb)
        labels.append(row['label'])
        
    return np.array(features), np.array(labels), ['R_mean', 'G_mean', 'B_mean']