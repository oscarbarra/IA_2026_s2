import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def ejecutar_exploracion(df_train):
    print("--- 3.1 Exploración de los Datos ---")
    rgb_ripe, rgb_unripe = [], []
    
    for _, row in df_train.iterrows():
        img = np.array(Image.open(row['img_path']).convert('RGB')) / 255.0
        mask = np.array(Image.open(row['mask_path']).convert('L')) > 128
        pixels = img[mask]
        (rgb_ripe if row['label'] == 1 else rgb_unripe).append(pixels)
            
    rgb_ripe, rgb_unripe = np.vstack(rgb_ripe), np.vstack(rgb_unripe)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for i, col in enumerate(['Red', 'Green', 'Blue']):
        axes[i].hist(rgb_ripe[:, i], bins=30, alpha=0.5, label='Maduro', color=col.lower())
        axes[i].hist(rgb_unripe[:, i], bins=30, alpha=0.5, label='Inmaduro', color='gray')
        axes[i].set_title(f'Canal {col}')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(f'{os.getcwd()}/Scripts/Imagenes/exploracion_canales.png')
    plt.close()