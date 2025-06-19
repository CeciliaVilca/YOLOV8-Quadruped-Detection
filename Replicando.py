import os
import json
import urllib.request
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import random
import matplotlib.pyplot as plt

# Importar librerías para ResNet50 y MobileNet
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image as keras_image
import tensorflow as tf

# Importar librerías para LBP
from skimage.feature import local_binary_pattern

# Importar librerías para segmentación con Superpíxeles y K-Means
from skimage.segmentation import slic
from sklearn.cluster import MiniBatchKMeans

# Importar librerías para clasificación
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Importar YOLO para detección en cascada
from ultralytics import YOLO

# Importaciones para multiprocessing
from multiprocessing import Pool, cpu_count

# ----------------------------------------------------
# --- DEFINICIONES DE FUNCIONES Y VARIABLES GLOBALES ---
# ----------------------------------------------------

# Estas variables se inicializan a None y serán cargadas en el bloque main.
# Son globales por su definición en este ámbito.
base_model_resnet = None
base_model_darknet_like = None

def apply_clahe_color(image_path, output_path):
    """
    Aplica la Ecualización Adaptativa del Histograma con Limitación de Contraste (CLAHE)
    a una imagen en color y la guarda.
    """
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


def _process_segmentation_image(args):
    """
    Función auxiliar para procesar la segmentación de una imagen
    usando SLIC y K-Means. Destinada a ser usada con multiprocessing.Pool.
    """
    image_path, output_path, n_segments, compactness, n_clusters = args
    img = cv2.imread(image_path)
    if img is None:
        return f"Advertencia: No se pudo cargar la imagen para segmentación: {image_path}"

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    try:
        segments = slic(img_lab, n_segments=n_segments, compactness=compactness, enforce_connectivity=True, sigma=1)
    except Exception as e:
        return f"Error SLIC en {image_path}: {e}"

    flattened_lab_colors = []
    unique_segments = np.unique(segments)
    if len(unique_segments) == 0:
        cv2.imwrite(output_path, img)
        return f"Advertencia: No hay superpíxeles válidos para {image_path}. Guardando imagen original."

    for segVal in unique_segments:
        mask = segments == segVal
        if np.any(mask):
            avg_color_lab = np.mean(img_lab[mask], axis=0)
            flattened_lab_colors.append(avg_color_lab)
    
    flattened_lab_colors = np.array(flattened_lab_colors, dtype="float32")
    
    n_clusters_actual = min(n_clusters, len(flattened_lab_colors))
    if n_clusters_actual == 0:
        cv2.imwrite(output_path, img)
        return f"Advertencia: No hay suficientes superpíxeles para crear {n_clusters} clusters en {image_path}. Guardando imagen original."

    try:
        clt = MiniBatchKMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
        labels = clt.fit_predict(flattened_lab_colors)
    except Exception as e:
        return f"Error K-Means en {image_path}: {e}"

    segmented_image = np.zeros_like(img)
    for (i, segVal) in enumerate(unique_segments):
        mask = segments == segVal
        if np.any(mask):
            cluster_center_lab = clt.cluster_centers_[labels[i]]
            segmented_image[mask] = cv2.cvtColor(np.uint8([[cluster_center_lab]]), cv2.COLOR_LAB2BGR)[0][0]
    
    cv2.imwrite(output_path, segmented_image)
    return None


def extract_lbp_features(image_path):
    """
    Extrae características LBP (Local Binary Pattern) de una imagen.
    """
    img_color = cv2.imread(image_path)
    if img_color is None:
        return None
    try:
        if len(img_color.shape) == 3:
            img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        else:
            img = img_color
        
        if img.shape[0] == 0 or img.shape[1] == 0:
            return None

        lbp = local_binary_pattern(img, P=24, R=3, method="uniform")
        n_bins = int(lbp.max() + 1) if lbp.max() > 0 else 2
        hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
        return hist.tolist()
    except Exception as e:
        print(f"DEBUG: LBP: Error inesperado al extraer LBP de {image_path}: {e}")
        return None


def extract_resnet_features(image_path):
    """
    Extrae características de una imagen utilizando el modelo ResNet50.
    Requiere que base_model_resnet esté cargado globalmente.
    """
    if base_model_resnet is None:
        return None
    try:
        img = keras_image.load_img(image_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = base_model_resnet.predict(img_array, verbose=0)
        return features.flatten().tolist()
    except Exception as e:
        print(f"DEBUG: ResNet: Error inesperado al extraer ResNet features de {image_path}: {e}")
        return None


def extract_darknet_like_features(image_path):
    """
    Extrae características de una imagen utilizando el modelo MobileNet
    como aproximación a Darknet19.
    Requiere que base_model_darknet_like esté cargado globalmente.
    """
    if base_model_darknet_like is None:
        return None
    try:
        img = keras_image.load_img(image_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = base_model_darknet_like.predict(img_array, verbose=0)
        return features.flatten().tolist()
    except Exception as e:
        print(f"DEBUG: Darknet-like: Error inesperado al extraer características de {image_path}: {e}")
        return None


# ----------------------------------------------------
# --- CÓDIGO PRINCIPAL (DENTRO DEL MAIN GUARD) ---
# ----------------------------------------------------

if __name__ == '__main__':
    # Configurar TensorFlow para evitar problemas de memoria en la GPU si es posible
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("No se detectaron GPUs o no están configuradas para TensorFlow.")


    # ==============================================================================
    # --- Paso 1: Descargar imágenes filtradas ---
    # ==============================================================================
    print("\n=============================================")
    print("--- Paso 1: Descargar imágenes filtradas ---")
    print("=============================================")

    download_dir = "ena24/images"
    metadata_file_path = "ena24/annotations/metadata.json"
    
    # Verificar si el directorio de imágenes ya tiene contenido suficiente
    # para considerar que la descarga ya se realizó
    min_images_for_download_check = 1900 # Si esperamos 2000, 1900 es un buen umbral
    
    if os.path.exists(download_dir) and len(os.listdir(download_dir)) >= min_images_for_download_check:
        print(f"Detectadas al menos {min_images_for_download_check} imágenes en {download_dir}. Saltando descarga de imágenes.")
        # Asumir que la metadata también está presente si las imágenes ya lo están
        if not os.path.exists(metadata_file_path):
             print(f"Advertencia: Metadata no encontrada en {metadata_file_path}, intentando descargarla.")
             # Fallback para descargar metadata si solo faltara eso
             metadata_url = "https://storage.googleapis.com/public-datasets-lila/ena24/ena24.json"
             os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)
             try:
                 urllib.request.urlretrieve(metadata_url, metadata_file_path)
                 print("Metadata descargada.")
             except Exception as e:
                 print(f"Error al descargar metadata.json: {e}. No se podrá continuar sin este archivo.")
                 exit()
    else:
        cuadrupedo_ids = {
            2, 3, 5, 6, 7, 10, 11, 12, 13,
            14, 15, 16, 19, 20, 21, 22
        }

        os.makedirs("ena24/annotations", exist_ok=True)

        metadata_url = "https://storage.googleapis.com/public-datasets-lila/ena24/ena24.json"
        
        if not os.path.exists(metadata_file_path):
            print(f"Descargando {os.path.basename(metadata_file_path)}...")
            try:
                urllib.request.urlretrieve(metadata_url, metadata_file_path)
                print("Metadata descargada.")
            except Exception as e:
                print(f"Error al descargar metadata.json: {e}. No se podrá continuar sin este archivo.")
                exit()
        else:
            print("Metadata.json ya existe.")

        with open(metadata_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_to_cats = defaultdict(set)
        for ann in data["annotations"]:
            if ann["category_id"] in cuadrupedo_ids:
                img_to_cats[ann["image_id"]].add(ann["category_id"])

        filtered_images = [img for img in data["images"] if img["id"] in img_to_cats]
        print(f"Se encontraron {len(filtered_images)} imágenes con cuadrúpedos.")

        base_url = "https://lilawildlife.blob.core.windows.net/lila-wildlife/ena24/images/"
        os.makedirs(download_dir, exist_ok=True)

        print(f"Descargando imágenes con cuadrúpedos en {download_dir}...")
        for img in tqdm(filtered_images[:300]):
            file_name = img["file_name"]
            url = base_url + file_name
            save_path = os.path.join(download_dir, file_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if not os.path.exists(save_path):
                retries = 3
                for attempt in range(retries):
                    try:
                        urllib.request.urlretrieve(url, save_path)
                        break
                    except Exception as e:
                        print(f"Error al descargar {file_name} (Intento {attempt + 1}/{retries}): {e}")
                else:
                    print(f"Error persistente al descargar {file_name}. Saltando.")
                    continue

                test_img = cv2.imread(save_path)
                if test_img is None:
                    print(f"Advertencia: Archivo descargado {file_name} no es una imagen válida. Eliminando.")
                    os.remove(save_path)
            else:
                pass

        print(f"Descarga de imágenes completada. Descargadas: {len([f for f in os.listdir(download_dir) if os.path.isfile(os.path.join(download_dir, f))])}")

    original_images_dir = "ena24/images"
    ahe_color_images_dir = "ena24/images_ahe_color"

    # ==============================================================================
    # --- Paso 2: Aplicar CLAHE a las imágenes descargadas ---
    # ==============================================================================

    print("\n===========================================")
    print("--- Paso 2: Aplicar CLAHE a las imágenes ---")
    print("===========================================")

    output_dir_ahe_color = "ena24/images_ahe_color"
    os.makedirs(output_dir_ahe_color, exist_ok=True)

    image_files_clahe = []
    for root, _, files in os.walk(download_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, download_dir)
                image_files_clahe.append(rel_path)

    # Verificar si todas las imágenes CLAHE ya existen
    all_clahe_exist = True
    if image_files_clahe: # Solo si hay imágenes para procesar
        for rel_path in image_files_clahe:
            output_path = os.path.join(output_dir_ahe_color, rel_path)
            if not os.path.exists(output_path):
                all_clahe_exist = False
                break
    else: # Si no hay imágenes descargadas, no hay CLAHE para hacer
        all_clahe_exist = False

    if all_clahe_exist and len(os.listdir(output_dir_ahe_color)) == len(image_files_clahe):
        print(f"Todas las imágenes CLAHE ya existen en {output_dir_ahe_color}. Saltando aplicación de CLAHE.")
    else:
        print(f"Aplicando CLAHE a {len(image_files_clahe)} imágenes a color...")
        for rel_path in tqdm(image_files_clahe):
            input_path = os.path.join(download_dir, rel_path)
            output_path = os.path.join(output_dir_ahe_color, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if not os.path.exists(output_path):
                apply_clahe_color(input_path, output_path)
            else:
                pass

        print(f"Preprocesamiento (CLAHE) completado. Las imágenes están en: {output_dir_ahe_color}")

    # ==============================================================================
    # --- Paso 3: Segmentación Intermedia con Superpíxeles y K-Means ---
    # ==============================================================================

    print("\n========================================================")
    print("--- Paso 3: Segmentación Intermedia (Superpíxeles + K-Means) ---")
    print("========================================================")

    output_dir_segmented = "ena24/images_segmented_kmeans"
    os.makedirs(output_dir_segmented, exist_ok=True)

    images_for_segmentation = [f for f in os.listdir(ahe_color_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Verificar si todas las imágenes segmentadas ya existen
    all_segmented_exist = True
    if images_for_segmentation:
        for filename in images_for_segmentation:
            output_path = os.path.join(output_dir_segmented, filename)
            if not os.path.exists(output_path):
                all_segmented_exist = False
                break
    else:
        all_segmented_exist = False

    if all_segmented_exist and len(os.listdir(output_dir_segmented)) == len(images_for_segmentation):
        print(f"Todas las imágenes segmentadas ya existen en {output_dir_segmented}. Saltando segmentación.")
    else:
        print(f"Aplicando segmentación intermedia a {len(images_for_segmentation)} imágenes desde {ahe_color_images_dir}...")

        tasks = []
        for filename in images_for_segmentation:
            input_path = os.path.join(ahe_color_images_dir, filename)
            output_path = os.path.join(output_dir_segmented, filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if not os.path.exists(output_path): # Solo añadir a la cola si no existe
                tasks.append((input_path, output_path, 200, 10, 5))

        if tasks:
            num_processes = cpu_count()
            print(f"Iniciando segmentación con {num_processes} procesos (multiprocessing)...")
            with Pool(processes=num_processes) as pool:
                for result in tqdm(pool.imap_unordered(_process_segmentation_image, tasks), total=len(tasks)):
                    if result:
                        print(result)
        else:
            print("Todas las imágenes de segmentación ya existen o no hay imágenes nuevas para procesar.")

        print(f"Segmentación intermedia completada. Imágenes segmentadas en: {output_dir_segmented}")

    segmented_images_dir = "ena24/images_segmented_kmeans"

    # ==============================================================================
    # --- Paso 4: Extracción de Características (ResNet50, Darknet19/MobileNet y LBP) ---
    # ==============================================================================

    print("\n=============================================")
    print("--- Paso 4: Extracción de Características ---")
    print("=============================================")

    input_dir_for_features = segmented_images_dir
    output_dir_features = "ena24/features"
    os.makedirs(output_dir_features, exist_ok=True)
    output_features_file = os.path.join(output_dir_features, "image_features.json")

    image_features = {}
    
    # Verificar si el archivo de características ya existe y es válido
    if os.path.exists(output_features_file) and os.path.getsize(output_features_file) > 0:
        try:
            with open(output_features_file, 'r') as f:
                image_features = json.load(f)
            # También verificar si el número de características cargadas es el esperado
            images_in_input_dir = [f for f in os.listdir(input_dir_for_features) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(image_features) == len(images_in_input_dir) and len(image_features) > 0:
                print(f"Características ya existen y cargadas desde: {output_features_file}. Saltando extracción.")
            else:
                print(f"Advertencia: El archivo de características {output_features_file} está incompleto o no coincide con las imágenes de entrada. Re-extrayendo características.")
                image_features = {}
        except json.JSONDecodeError:
            print(f"Advertencia: El archivo de características {output_features_file} está corrupto. Re-extrayendo características.")
            image_features = {}
        except Exception as e:
            print(f"Error al cargar características existentes: {e}. Re-extrayendo características.")
            image_features = {}

    if not image_features: # Si las características no se cargaron o están incompletas/corruptas
        print("Cargando modelo ResNet50...")
        try:
            # ¡CORRECCIÓN AQUÍ! Elimina la palabra 'global'
            base_model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            print("ResNet50 cargado exitosamente.")
        except Exception as e:
            print(f"Error al cargar ResNet50: {e}")
            base_model_resnet = None
            print("Continuando sin características ResNet50 si el modelo no pudo cargarse.")

        print("Cargando modelo para características Darknet (usando MobileNet como aproximación)...")
        try:
            # ¡CORRECCIÓN AQUÍ! Elimina la palabra 'global'
            base_model_darknet_like = MobileNet(weights='imagenet', include_top=False, pooling='avg')
            print("MobileNet cargado exitosamente como aproximación a Darknet.")
        except Exception as e:
            print(f"Error al cargar MobileNet: {e}")
            base_model_darknet_like = None
            print("Continuando sin características Darknet si el modelo no pudo cargarse.")

        images_to_extract_features = [f for f in os.listdir(input_dir_for_features) if os.path.exists(os.path.join(input_dir_for_features, f))]
        
        if not images_to_extract_features:
            print(f"DEBUG: La lista de imágenes para extraer características de {input_dir_for_features} está vacía. Verifica el Paso 3.")
                
        print(f"Extrayendo características ResNet50, Darknet (aproximación) y LBP de {len(images_to_extract_features)} imágenes desde {input_dir_for_features}...")
        for filename in tqdm(images_to_extract_features):
            img_path = os.path.join(input_dir_for_features, filename)
            
            current_image_feats = {}

            lbp_feat = extract_lbp_features(img_path)
            if lbp_feat is not None:
                current_image_feats['lbp'] = lbp_feat

            if base_model_resnet:
                resnet_feat = extract_resnet_features(img_path)
                if resnet_feat is not None:
                    current_image_feats['resnet'] = resnet_feat
            
            if base_model_darknet_like:
                darknet_feat = extract_darknet_like_features(img_path)
                if darknet_feat is not None:
                    current_image_feats['darknet'] = darknet_feat

            if not current_image_feats:
                print(f"DEBUG: Ninguna característica (LBP, ResNet, Darknet) pudo ser extraída para {filename}.")
                    
            if current_image_feats:
                image_features[filename] = current_image_feats

        print("Extracción de características completada.")

        try:
            with open(output_features_file, 'w') as f:
                json.dump(image_features, f, indent=4)
            print(f"Características guardadas en: {output_features_file}")
        except Exception as e:
            print(f"Error al guardar las características en JSON: {e}")
    
    print("\n¡El script ha terminado de ejecutar los pasos de preprocesamiento, segmentación y extracción de características!")
    print(f"Puedes encontrar las características guardadas en '{output_features_file}'.")


    # ==============================================================================
    # --- Paso 5: Clasificación Inicial y Filtrado para YOLOv8 ---
    # ==============================================================================
    print("\n=============================================")
    print("--- Paso 5: Clasificación Inicial y Filtrado ---")
    print("=============================================")

    features_file_path = "ena24/features/image_features.json"
    output_filtered_list_path = os.path.join("ena24", "images_for_yolo.json")

    # Verificar si el archivo de la lista filtrada ya existe y es válido
    if os.path.exists(output_filtered_list_path) and os.path.getsize(output_filtered_list_path) > 0:
        try:
            with open(output_filtered_list_path, 'r') as f:
                images_for_yolo = json.load(f)
            if len(images_for_yolo) > 0:
                print(f"Lista de imágenes filtradas para YOLOv8 ya existe y fue cargada desde: {output_filtered_list_path}. Saltando clasificación.")
            else:
                print(f"Advertencia: La lista de imágenes filtradas {output_filtered_list_path} está vacía. Re-ejecutando clasificación.")
                images_for_yolo = []
        except json.JSONDecodeError:
            print(f"Advertencia: La lista de imágenes filtradas {output_filtered_list_path} está corrupta. Re-ejecutando clasificación.")
            images_for_yolo = []
    else: # Ejecutar clasificación si el archivo no existe o está vacío
        try:
            with open(features_file_path, 'r') as f:
                loaded_features = json.load(f)
            print(f"Características cargadas desde: {features_file_path}")
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo de características en {features_file_path}. Asegúrate de haber ejecutado el Paso 4.")
            exit()
        except json.JSONDecodeError:
            print(f"Error: No se pudo decodificar el JSON de características. El archivo podría estar corrupto.")
            exit()

        X = []
        y = []
        image_filenames_for_classification = []

        for filename, feats in loaded_features.items():
            lbp_vec = np.array(feats.get('lbp', []))
            resnet_vec = np.array(feats.get('resnet', []))
            darknet_vec = np.array(feats.get('darknet', []))

            combined_features = []
            if lbp_vec.size > 0:
                combined_features.extend(lbp_vec)
            if resnet_vec.size > 0:
                combined_features.extend(resnet_vec)
            if darknet_vec.size > 0:
                combined_features.extend(darknet_vec)

            if len(combined_features) > 0:
                X.append(np.array(combined_features))
                y.append(1)
                image_filenames_for_classification.append(filename)
            else:
                print(f"Advertencia: Saltando imagen {filename} en el clasificador debido a características faltantes o vacías.")

        X = np.array(X)
        y = np.array(y)

        if X.shape[0] == 0:
            print("Error: No se encontraron características válidas para entrenar el clasificador. La lista de imágenes para YOLO estará vacía.")
            images_for_yolo = []
        else:
            print(f"Número de muestras de características cargadas: {X.shape[0]}")
            print(f"Dimensiones de cada vector de características: {X.shape[1]}")

            stratify_param = y if len(np.unique(y)) > 1 else None
            X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
                X, y, image_filenames_for_classification, test_size=0.2, random_state=42, stratify=stratify_param
            )

            print("Entrenando el clasificador Random Forest...")
            classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            classifier.fit(X_train, y_train)
            print("Clasificador entrenado.")

            y_pred = classifier.predict(X_test)

            print("\nResultados del Clasificador Inicial:")
            print(f"Precisión (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
            print("Reporte de Clasificación:")
            if len(np.unique(y_test)) > 1:
                print(classification_report(y_test, y_pred, target_names=['No Animal', 'Animal']))
            else:
                print("Solo una clase presente en el conjunto de prueba, no se puede generar un reporte de clasificación completo.")

            print("\nSimulando la fase de filtrado para YOLOv8...")
            predicted_labels_all_images = classifier.predict(X)

            images_for_yolo = [
                filename for i, filename in enumerate(image_filenames_for_classification)
                if predicted_labels_all_images[i] == 1
            ]

            print(f"De {len(image_filenames_for_classification)} imágenes, {len(images_for_yolo)} imágenes serán pasadas a YOLOv8.")

        try:
            with open(output_filtered_list_path, 'w') as f:
                json.dump(images_for_yolo, f, indent=4)
            print(f"Lista de imágenes filtradas para YOLOv8 guardada en: {output_filtered_list_path}")
        except Exception as e:
            print(f"Error al guardar la lista de imágenes filtradas en JSON: {e}")

    print("\n¡El script ha completado la clasificación inicial y el filtrado para YOLOv8!")


    # ==============================================================================
    # --- Paso 6: Detección con YOLOv8 en Cascada ---
    # ==============================================================================

    print("\n=============================================")
    print("--- Paso 6: Detección con YOLOv8 en Cascada ---")
    print("=============================================")

    input_dir_for_yolo_inference = ahe_color_images_dir
    output_dir_cascaded_detections = "ena24/cascaded_detections"
    os.makedirs(output_dir_cascaded_detections, exist_ok=True)
    output_cascaded_detections_file = os.path.join(output_dir_cascaded_detections, "yolov8_cascaded_detections.json")

    images_to_process_with_yolo = []
    # Cargar la lista de imágenes que fueron filtradas por el clasificador
    try:
        with open(output_filtered_list_path, 'r') as f:
            images_to_process_with_yolo = json.load(f)
        print(f"Cargadas {len(images_to_process_with_yolo)} imágenes para procesar con YOLOv8 Cascaded.")
    except FileNotFoundError:
        print(f"Error: No se encontró la lista de imágenes filtradas en {output_filtered_list_path}. No se puede continuar con YOLOv8.")
        images_to_process_with_yolo = []
    except json.JSONDecodeError:
        print(f"Error: El archivo de imágenes para YOLO {output_filtered_list_path} está corrupto. Creando lista vacía.")
        images_to_process_with_yolo = []


    # Verificar si el archivo de resultados de detección final ya existe y es válido
    if os.path.exists(output_cascaded_detections_file) and os.path.getsize(output_cascaded_detections_file) > 0:
        try:
            with open(output_cascaded_detections_file, 'r') as f:
                existing_detections = json.load(f)
            # Contar cuántas imágenes de la lista filtrada ya tienen detecciones en el JSON
            processed_count = sum(1 for img_name in images_to_process_with_yolo if img_name in existing_detections)
            if processed_count == len(images_to_process_with_yolo) and len(images_to_process_with_yolo) > 0:
                print(f"Todas las imágenes ({processed_count}) ya tienen resultados de detección en {output_cascaded_detections_file}. Saltando detección YOLOv8.")
                # Si todo está procesado, podemos cargar los resultados existentes
                cascaded_detection_results = existing_detections
            else:
                print(f"Advertencia: El archivo de detección {output_cascaded_detections_file} está incompleto ({processed_count}/{len(images_to_process_with_yolo)}). Reanudando detección.")
                cascaded_detection_results = existing_detections # Cargar lo que ya hay
        except json.JSONDecodeError:
            print(f"Advertencia: El archivo de detección {output_cascaded_detections_file} está corrupto. Re-ejecutando detección.")
            cascaded_detection_results = {} # Reiniciar si hay corrupción
    else:
        cascaded_detection_results = {} # Inicializar vacío si no existe el archivo

    # Solo ejecutar YOLO si hay imágenes pendientes de procesamiento
    images_pending_yolo = [img for img in images_to_process_with_yolo if img not in cascaded_detection_results]
    
    if not images_pending_yolo:
        if len(images_to_process_with_yolo) == 0:
            print("No hay imágenes filtradas por el clasificador inicial para pasar a YOLOv8 Cascaded.")
        else:
            print("Todas las imágenes filtradas ya han sido procesadas por YOLOv8 Cascaded.")
        print("El proceso de detección de YOLOv8 Cascaded se saltará.")
    else:
        print("Cargando modelos YOLOv8 para la cascada (yolov8n.pt para grueso, yolov8s.pt para refinado)...")
        try:
            model_coarse = YOLO('yolov8n.pt')
            model_refined = YOLO('yolov8s.pt')
            print("Modelos YOLOv8 de cascada cargados exitosamente.")
        except Exception as e:
            print(f"Error al cargar modelos YOLOv8: {e}")
            print("Asegúrate de haber instalado 'ultralytics' y de tener conexión a internet para la descarga inicial de los modelos.")
            exit()

        print(f"Iniciando la detección de objetos con YOLOv8 en Cascada en {len(images_pending_yolo)} imágenes (pendientes)...")

        for filename in tqdm(images_pending_yolo):
            img_path_original = os.path.join(input_dir_for_yolo_inference, filename)

            if not os.path.exists(img_path_original):
                print(f"Advertencia: La imagen original {img_path_original} no existe. Saltando.")
                cascaded_detection_results[filename] = [] # Registrar como no procesada o sin detections
                continue

            try:
                img_to_draw_on = cv2.imread(img_path_original)
                if img_to_draw_on is None:
                    print(f"Advertencia: No se pudo cargar la imagen para dibujar {img_path_original}. Saltando visualización para esta imagen.")
                    cascaded_detection_results[filename] = []
                    continue

                current_image_detections = []

                results_coarse = model_coarse(img_path_original, conf=0.25, iou=0.5, verbose=False)

                rois_for_refinement = []
                for r_coarse in results_coarse:
                    boxes_coarse = r_coarse.boxes
                    for box_coarse in boxes_coarse:
                        x1, y1, x2, y2 = map(int, box_coarse.xyxy[0].tolist())
                        confidence = float(box_coarse.conf[0])
                        class_id = int(box_coarse.cls[0])
                        
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img_to_draw_on.shape[1], x2)
                        y2 = min(img_to_draw_on.shape[0], y2)

                        if x2 > x1 and y2 > y1:
                            rois_for_refinement.append({
                                "box": [x1, y1, x2, y2],
                                "confidence": confidence,
                                "class_id": class_id
                            })

                if rois_for_refinement:
                    final_detections = []
                    for roi_info in rois_for_refinement:
                        x1_roi, y1_roi, x2_roi, y2_roi = roi_info["box"]
                        
                        roi_img = img_to_draw_on[y1_roi:y2_roi, x1_roi:x2_roi].copy()

                        if roi_img.shape[0] == 0 or roi_img.shape[1] == 0:
                            continue

                        results_refined = model_refined(roi_img, conf=0.5, iou=0.7, verbose=False)

                        for r_refined in results_refined:
                            boxes_refined = r_refined.boxes
                            names_refined = r_refined.names
                            for box_refined in boxes_refined:
                                x1_rel, y1_rel, x2_rel, y2_rel = map(int, box_refined.xyxy[0].tolist())
                                confidence_refined = float(box_refined.conf[0])
                                class_id_refined = int(box_refined.cls[0])
                                class_name_refined = names_refined[class_id_refined]

                                x1_abs = x1_rel + x1_roi
                                y1_abs = y1_rel + y1_roi
                                x2_abs = x2_rel + x1_roi
                                y2_abs = y2_rel + y1_roi

                                x1_abs = max(0, x1_abs)
                                y1_abs = max(0, y1_abs)
                                x2_abs = min(img_to_draw_on.shape[1], x2_abs)
                                y2_abs = min(img_to_draw_on.shape[0], y2_abs)

                                if x2_abs > x1_abs and y2_abs > y1_abs:
                                    final_detections.append({
                                        "box": [x1_abs, y1_abs, x2_abs, y2_abs],
                                        "confidence": confidence_refined,
                                        "class_id": class_id_refined,
                                        "class_name": class_name_refined
                                    })
                        
                    final_detections.sort(key=lambda x: x["confidence"], reverse=True)

                    for det in final_detections:
                        x1, y1, x2, y2 = det["box"]
                        confidence = det["confidence"]
                        class_name = det["class_name"]

                        current_image_detections.append(det)

                        color = (0, 255, 0)
                        cv2.rectangle(img_to_draw_on, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name}: {confidence:.2f}"
                        text_y = max(y1 - 10, 15)
                        cv2.putText(img_to_draw_on, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                    cascaded_detection_results[filename] = current_image_detections

                    output_detection_img_path = os.path.join(output_dir_cascaded_detections, filename)
                    os.makedirs(os.path.dirname(output_detection_img_path), exist_ok=True)
                    cv2.imwrite(output_detection_img_path, img_to_draw_on)
                else:
                    cascaded_detection_results[filename] = [] # Registrar como sin detecciones

            except Exception as e:
                print(f"Error procesando imagen {filename} con YOLOv8 Cascaded: {e}")
                cascaded_detection_results[filename] = [] # Registrar con un error
                continue

        # Guardar todos los resultados de detección en un archivo JSON (incluyendo los nuevos)
        try:
            with open(output_cascaded_detections_file, 'w') as f:
                json.dump(cascaded_detection_results, f, indent=4)
            print(f"Resultados de detección de YOLOv8 Cascaded guardados/actualizados en: {output_cascaded_detections_file}")
        except Exception as e:
            print(f"Error al guardar los resultados de detección en JSON: {e}")


    print("\n¡El script ha completado todos los pasos, incluyendo la detección en cascada con YOLOv8!")
    print("Puedes revisar las imágenes con detecciones y el archivo JSON de resultados en la carpeta 'ena24/cascaded_detections'.")

    