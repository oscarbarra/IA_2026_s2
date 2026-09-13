
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def ejecutar_exploracion(df_train):
    """
    Analiza canales RGB y HSV diferenciando fruto vs fondo
    y frutos maduros vs inmaduros sobre los datos de entrenamiento.
    """
    print("--- 3.1 Exploración de los Datos ---")
    rgb_ripe, rgb_unripe = [], []
    
    for _, row in df_train.iterrows():
        img = np.array(Image.open(row['img_path']).convert('RGB')) / 255.0
        mask = np.array(Image.open(row['mask_path']).convert('L')) > 128
        pixels = img[mask]
        
        if row['label'] == 1:
            rgb_ripe.append(pixels)
        else:
            rgb_unripe.append(pixels)
            
    rgb_ripe = np.vstack(rgb_ripe)
    rgb_unripe = np.vstack(rgb_unripe)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['Red', 'Green', 'Blue']
    for i, col in enumerate(colors):
        axes[i].hist(rgb_ripe[:, i], bins=30, alpha=0.5, label='Maduro', color=col.lower())
        axes[i].hist(rgb_unripe[:, i], bins=30, alpha=0.3, label='Inmaduro', color='gray')
        axes[i].set_title(f'Canal {col}')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig('exploracion_canales.png')
    plt.close()
    print("Gráfico 'exploracion_canales.png' generado.")