import numpy as np
import cv2
import os

# Carrega o volume
volume = np.load('volumes_3d/demente_leve_grupo0.npy')  # substitua pelo caminho certo

# Cria diretório de saída
output_dir = 'slices_png'
os.makedirs(output_dir, exist_ok=True)

# Salva cada fatia como PNG
for i in range(volume.shape[0]):
    slice_img = volume[i]
    normalized = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX)
    img_uint8 = normalized.astype(np.uint8)
    cv2.imwrite(f"{output_dir}/slice_{i:03}.png", img_uint8)

print(f"Salvo {volume.shape[0]} fatias em: {output_dir}")
