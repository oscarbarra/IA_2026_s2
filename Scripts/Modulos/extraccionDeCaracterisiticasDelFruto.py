
import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv
from Modulos.segmentacionMedianteKmeans import segmentar_imagen

def extraer_caracteristicas_dataset(df, best_channels, seed=42):
    """
    Extrae promedios [R, G, B, H, S, V] sobre los píxeles de la máscara segmentada.
    """
    print("--- 3.3 Extracción de Características del Fruto ---")
    features = []
    labels = []
    
    for _, row in df.iterrows():
        img_pil = Image.open(row['img_path']).convert('RGB')
        img_np = np.array(img_pil) / 255.0
        
        mask_pred = segmentar_imagen(img_np, best_channels, seed=seed)
        
        if not np.any(mask_pred):
            # Fallback si no detecta píxeles
            mask_pred = np.ones((img_np.shape[0], img_np.shape[1]), dtype=bool)
            
        fruit_pixels_rgb = img_np[mask_pred]
        fruit_pixels_hsv = rgb_to_hsv(fruit_pixels_rgb)
        
        mean_rgb = fruit_pixels_rgb.mean(axis=0)
        mean_hsv = fruit_pixels_hsv.mean(axis=0)
        
        feat_vector = np.concatenate([mean_rgb, mean_hsv])
        features.append(feat_vector)
        labels.append(row['label'])
        
    feature_names = ['R_mean', 'G_mean', 'B_mean', 'H_mean', 'S_mean', 'V_mean']
    return np.array(features), np.array(labels), feature_names