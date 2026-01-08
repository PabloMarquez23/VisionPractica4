import cv2
import os
import numpy as np
from shutil import copyfile

# RUTA CORREGIDA según tu terminal
dataset_path = '/home/pablomar/Escritorio/Practica_Vision/supervisely_person_clean_2667_img/supervisely_person_clean_2667_img/'
dest_images = './datasets/person_data/images/'
dest_labels = './datasets/person_data/labels/'

# Crear carpetas si no existen
os.makedirs(dest_images, exist_ok=True)
os.makedirs(dest_labels, exist_ok=True)

def mask_to_yolo_polygons(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return []
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for cnt in contours:
        if len(cnt) > 4:
            poly = cnt.reshape(-1).tolist()
            poly_norm = [x / (mask.shape[1] if i % 2 == 0 else mask.shape[0]) for i, x in enumerate(poly)]
            polygons.append(poly_norm)
    return polygons

print("Iniciando conversión... esto puede tardar un poco.")

# El dataset de Kaggle suele tener carpetas 'images' y 'masks'
images_dir = os.path.join(dataset_path, 'images')
masks_dir = os.path.join(dataset_path, 'masks')

count = 0
for filename in os.listdir(images_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        # Copiar imagen a la estructura del proyecto
        copyfile(os.path.join(images_dir, filename), os.path.join(dest_images, filename))
        
        # Procesar máscara correspondiente
        mask_path = os.path.join(masks_dir, filename) # En este dataset el nombre coincide
        
        if os.path.exists(mask_path):
            polygons = mask_to_yolo_polygons(mask_path)
            txt_name = os.path.splitext(filename)[0] + ".txt"
            with open(os.path.join(dest_labels, txt_name), 'w') as f:
                for poly in polygons:
                    # Clase 0 para 'person' como pide la guía
                    line = "0 " + " ".join([f"{coord:.6f}" for coord in poly])
                    f.write(line + "\n")
            count += 1

print(f"¡Hecho! Se procesaron {count} imágenes y etiquetas.")