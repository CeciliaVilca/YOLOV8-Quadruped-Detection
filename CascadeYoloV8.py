import os
import json
import urllib.request
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from skimage.segmentation import slic
from sklearn.cluster import MiniBatchKMeans

# --- Paso 1: Descargar imágenes filtradas ---

cuadrupedo_ids = {
    1, 2, 3, 5, 6, 7, 10, 11, 12, 13,
    14, 15, 16, 19, 20, 21, 22
}

with open("ena24/annotations/metadata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

img_to_cats = defaultdict(set)
for ann in data["annotations"]:
    if ann["category_id"] in cuadrupedo_ids:
        img_to_cats[ann["image_id"]].add(ann["category_id"])

filtered_images = [img for img in data["images"] if img["id"] in img_to_cats]
print(f"Se encontraron {len(filtered_images)} imágenes con cuadrúpedos.")

base_url = "https://lilawildlife.blob.core.windows.net/lila-wildlife/ena24/images/"
download_dir = "ena24/images"
os.makedirs(download_dir, exist_ok=True)

for img in tqdm(filtered_images[:100]):
    file_name = img["file_name"]
    url = base_url + file_name
    save_path = os.path.join(download_dir, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(save_path):
        try:
            urllib.request.urlretrieve(url, save_path)
        except Exception as e:
            print(f"Error al descargar {file_name}: {e}")

# --- Paso 2: Aplicar CLAHE a las imágenes descargadas ---

def apply_clahe_color(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Advertencia: No se pudo cargar la imagen {image_path}")
        return

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    img_clahe_lab = cv2.merge([l_clahe, a, b])
    img_clahe_bgr = cv2.cvtColor(img_clahe_lab, cv2.COLOR_LAB2BGR)

    cv2.imwrite(output_path, img_clahe_bgr)

output_dir_ahe_color = "ena24/images_ahe_color"
os.makedirs(output_dir_ahe_color, exist_ok=True)

image_files = []
for root, _, files in os.walk(download_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, download_dir)
            image_files.append(rel_path)

print(f"Aplicando CLAHE a {len(image_files)} imágenes a color...")
for rel_path in tqdm(image_files):
    input_path = os.path.join(download_dir, rel_path)
    output_path = os.path.join(output_dir_ahe_color, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(output_path):  # Solo procesa si no existe el archivo ya generado
        apply_clahe_color(input_path, output_path)
    else:
        pass  # Ya existe, se salta

print(f"Preprocesamiento (CLAHE) completado. Las imágenes están en: {output_dir_ahe_color}")

# --- Paso 3: Segmentación con SLIC + MiniBatchKMeans ---

def segment_image_slic_clustering(image_path, output_mask_path, n_segments=1000, compactness=0.5, n_clusters=5):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Advertencia: No se pudo cargar la imagen {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    segments = slic(img_rgb, n_segments=n_segments, compactness=compactness, sigma=0)

    features = []
    unique_segments_ids = np.unique(segments)
    for i in unique_segments_ids:
        mask = (segments == i)
        if np.sum(mask) > 0:
            mean_color = img_rgb[mask].mean(axis=0)
            features.append(mean_color)
        else:
            features.append(np.zeros(3))

    features = np.array(features)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(features)

    output_image = np.zeros_like(img_rgb)
    for i, label in zip(unique_segments_ids, labels):
        output_image[segments == i] = kmeans.cluster_centers_[label].astype(np.uint8)

    cv2.imwrite(output_mask_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

output_dir_segmentation = "ena24/images_segmented"
os.makedirs(output_dir_segmentation, exist_ok=True)

images_to_segment = []
for root, _, files in os.walk(output_dir_ahe_color):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, output_dir_ahe_color)
            images_to_segment.append(rel_path)

print(f"Aplicando segmentación Superpixels + clustering a {len(images_to_segment)} imágenes...")
for rel_path in tqdm(images_to_segment):
    input_path = os.path.join(output_dir_ahe_color, rel_path)
    output_path = os.path.join(output_dir_segmentation, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    segment_image_slic_clustering(input_path, output_path)

print("Segmentación completada. Las imágenes segmentadas están en:", output_dir_segmentation)
