import os
import json
import urllib.request
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
# from skimage.segmentation import slic # Ya no se usa
# from sklearn.cluster import MiniBatchKMeans # Ya no se usa
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- Para ResNet50 ---
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# --- Para LBP ---
from skimage.feature import local_binary_pattern

# Importar librerías para clasificación
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==============================================================================
# --- Paso 1: Descargar imágenes filtradas ---
# ==============================================================================

print("\n=============================================")
print("--- Paso 1: Descargar imágenes filtradas ---")
print("=============================================")

cuadrupedo_ids = {
    1, 2, 3, 5, 6, 7, 10, 11, 12, 13,
    14, 15, 16, 19, 20, 21, 22
}

# Crear directorio para anotaciones si no existe (asumimos metadata.json está allí)
os.makedirs("ena24/annotations", exist_ok=True)

# Descargar metadata.json si no existe (muy importante para este paso)
metadata_url = "https://lilawildlife.blob.core.windows.net/lila-wildlife/ena24/annotations/ena24.json"
metadata_file_path = "ena24/annotations/metadata.json"
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
download_dir = "ena24/images"
os.makedirs(download_dir, exist_ok=True)

print(f"Descargando hasta 100 imágenes con cuadrúpedos en {download_dir}...")
for img in tqdm(filtered_images[:100]): # Limitar a 100 imágenes como en el código original
    file_name = img["file_name"]
    url = base_url + file_name
    save_path = os.path.join(download_dir, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(save_path):
        try:
            urllib.request.urlretrieve(url, save_path)
        except Exception as e:
            print(f"Error al descargar {file_name}: {e}")
    else:
        pass # Ya existe, se salta

print(f"Descarga de imágenes completada. Descargadas: {len([f for f in os.listdir(download_dir) if os.path.isfile(os.path.join(download_dir, f))])}")


# ==============================================================================
# --- Paso 2: Aplicar CLAHE a las imágenes descargadas ---
# ==============================================================================

print("\n===========================================")
print("--- Paso 2: Aplicar CLAHE a las imágenes ---")
print("===========================================")

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

image_files_clahe = [] # Renombrado para mayor claridad
for root, _, files in os.walk(download_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, download_dir)
            image_files_clahe.append(rel_path)

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
# --- NOTA: El Paso 3 (Segmentación con SLIC + K-Means) ha sido omitido ---
# ==============================================================================

# -------------------- RUTAS (ajustadas a CLAHE) --------------------
# Ya no necesitamos el directorio de imágenes segmentadas
original_images_dir = "ena24/images"
ahe_color_images_dir = "ena24/images_ahe_color"

# -------------------- PARÁMETROS --------------------
num_images_to_display = 5 

# -------------------- LISTAR ARCHIVOS (ahora de CLAHE) --------------------
all_clahe_images = [f for f in os.listdir(ahe_color_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# --- Descomenta si quieres mostrar imágenes (Original, CLAHE) ---
"""
# -------------------- MOSTRAR IMÁGENES --------------------
if all_clahe_images:
    sample_images = random.sample(all_clahe_images, min(num_images_to_display, len(all_clahe_images)))

    print("\n Mostrando algunas imágenes para comparación (Original, CLAHE):")
    plt.figure(figsize=(12, 8)) # Ajustado para 2 filas en lugar de 3

    for i, img_filename in enumerate(sample_images):
        # ---------- IMAGEN ORIGINAL ----------
        img_path_orig = os.path.join(original_images_dir, img_filename)
        img_orig = cv2.imread(img_path_orig)
        if img_orig is not None:
            plt.subplot(2, num_images_to_display, i + 1)
            plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
            plt.title("Original", fontsize=10)
            plt.axis('off')

        # ---------- IMAGEN CLAHE ----------
        img_path_ahe = os.path.join(ahe_color_images_dir, img_filename)
        img_ahe = cv2.imread(img_path_ahe)
        if img_ahe is not None:
            plt.subplot(2, num_images_to_display, i + 1 + num_images_to_display)
            plt.imshow(cv2.cvtColor(img_ahe, cv2.COLOR_BGR2RGB))
            plt.title("CLAHE Color", fontsize=10)
            plt.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("No se encontraron imágenes CLAHE en el directorio:", ahe_color_images_dir)
"""

# ==============================================================================
# --- Paso 4: Extracción de Características (ResNet50 y LBP) ---
# ==============================================================================

print("\n=============================================")
print("--- Paso 4: Extracción de Características ---")
print("=============================================")

# --- Configuración de Directorios para Características ---
# ¡AHORA SE EXTRAEN LAS CARACTERÍSTICAS DIRECTAMENTE DE LAS IMÁGENES CLAHE!
input_dir_for_features = "ena24/images_ahe_color" 
output_dir_features = "ena24/features"
os.makedirs(output_dir_features, exist_ok=True)
output_features_file = os.path.join(output_dir_features, "image_features.json")

# --- Lógica para evitar re-extracción ---
image_features = {} 
if os.path.exists(output_features_file) and os.path.getsize(output_features_file) > 0:
    try:
        with open(output_features_file, 'r') as f:
            image_features = json.load(f)
        print(f"Características ya existen y cargadas desde: {output_features_file}")
    except json.JSONDecodeError:
        print(f"Advertencia: El archivo de características {output_features_file} está corrupto. Re-extrayendo características.")
        image_features = {} 
    except Exception as e:
        print(f"Error al cargar características existentes: {e}. Re-extrayendo características.")
        image_features = {} 

if not image_features: 
    # --- Cargar Modelo ResNet50 (solo si vamos a extraer) ---
    print("Cargando modelo ResNet50...")
    try:
        base_model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        print("ResNet50 cargado exitosamente.")
    except Exception as e:
        print(f"Error al cargar ResNet50: {e}")
        base_model_resnet = None 
        print("Continuando sin características ResNet50 si el modelo no pudo cargarse.")

    # --- Funciones de Extracción de Características ---
    # ¡LA INDENTACIÓN DE ESTAS FUNCIONES FUE CORREGIDA AQUÍ!
    def extract_lbp_features(image_path):
        """
        Extrae características de Local Binary Pattern (LBP) de una imagen.
        """
        img_color = cv2.imread(image_path) # Leer la imagen a color
        if img_color is None:
            print(f"DEBUG: LBP: No se pudo cargar la imagen (cv2.imread es None) para LBP: {image_path}")
            return None
        
        try:
            # Convertir a escala de grises explícitamente si es una imagen a color
            if len(img_color.shape) == 3: # Si tiene 3 canales (BGR)
                img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            else: # Si ya es escala de grises (1 canal o ya es 2D)
                img = img_color
            
            # Verificar si la imagen es válida (no vacía) después de la conversión
            if img.shape[0] == 0 or img.shape[1] == 0:
                print(f"DEBUG: LBP: Imagen vacía o corrupta (shape 0) para LBP: {image_path}")
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
        Extrae características de una imagen usando ResNet50 pre-entrenado.
        """
        if base_model_resnet is None:
            # Este mensaje ya se imprime al intentar cargar el modelo por primera vez.
            # print(f"DEBUG: ResNet: base_model_resnet es None para {image_path}") 
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

    # --- Procesamiento de Imágenes para Extracción de Características ---
    # Esta sección también debe estar al mismo nivel de indentación que las funciones 'def'
    images_to_extract_features = [f for f in image_files_clahe if os.path.exists(os.path.join(input_dir_for_features, f))]
    
    if not images_to_extract_features: # DIAGNOSTIC
        print(f"DEBUG: La lista de imágenes para extraer características de {input_dir_for_features} está vacía. Verifica el Paso 2.")
        
    print(f"Extrayendo características ResNet50 y LBP de {len(images_to_extract_features)} imágenes desde {input_dir_for_features}...")
    for filename in tqdm(images_to_extract_features):
        img_path = os.path.join(input_dir_for_features, filename)
        
        current_image_feats = {}

        # LBP
        lbp_feat = extract_lbp_features(img_path)
        if lbp_feat is not None:
            current_image_feats['lbp'] = lbp_feat 

        # ResNet50
        resnet_feat = extract_resnet_features(img_path)
        if resnet_feat is not None:
            current_image_feats['resnet'] = resnet_feat 

        if not current_image_feats: # DIAGNOSTIC: Check if both failed
            print(f"DEBUG: Ninguna característica (LBP o ResNet) pudo ser extraída para {filename}.")
            
        if current_image_feats:
            image_features[filename] = current_image_feats

    print("Extracción de características completada.")

    # --- Guardar Características ---
    try:
        with open(output_features_file, 'w') as f:
            json.dump(image_features, f, indent=4)
        print(f"Características guardadas en: {output_features_file}")
    except Exception as e:
        print(f"Error al guardar las características en JSON: {e}")
else:
    print("Saltando la extracción de características ya que el archivo ya existe.")

print("\n¡El script ha terminado de ejecutar todos los pasos hasta la extracción de características!")
print(f"Puedes encontrar las características guardadas en '{output_features_file}'.")

# --- Clasificador Inicial para Filtrado ---

# Directorio donde se guardaron las características
features_file_path = "ena24/features/image_features.json"

# Cargar las características
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

# Preparar los datos para el clasificador
X = [] 
y = [] 
image_filenames_for_classification = [] 

for filename, feats in loaded_features.items():
    lbp_vec = np.array(feats.get('lbp', []))
    resnet_vec = np.array(feats.get('resnet', []))

    if lbp_vec.size > 0 and resnet_vec.size > 0:
        combined_features = np.concatenate((lbp_vec, resnet_vec))
        X.append(combined_features)
        y.append(1) # Etiquetar como '1' (animal presente)
        image_filenames_for_classification.append(filename)
    else:
        print(f"Advertencia: Saltando imagen {filename} en el clasificador debido a características faltantes o vacías.")

X = np.array(X)
y = np.array(y)

if X.shape[0] == 0:
    print("Error: No se encontraron características válidas para entrenar el clasificador. La lista de imágenes para YOLO estará vacía.")
    # Si no hay datos, inicializamos una lista vacía para YOLO y continuamos
    images_for_yolo = [] 
else:
    print(f"Número de muestras de características cargadas: {X.shape[0]}")
    print(f"Dimensiones de cada vector de características: {X.shape[1]}")

    # Dividir los datos en conjuntos de entrenamiento y prueba
    # Asegúrate de que y_train tenga al menos 2 clases si vas a usar stratify.
    # Si solo hay una clase (y.shape[0] > 0 y len(np.unique(y)) == 1), stratify debe ser None.
    stratify_param = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X, y, image_filenames_for_classification, test_size=0.2, random_state=42, stratify=stratify_param
    )

    # Crear y entrenar un clasificador (RandomForestClassifier es una buena opción por su robustez)
    print("Entrenando el clasificador Random Forest...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) 
    classifier.fit(X_train, y_train)
    print("Clasificador entrenado.")

    # Evaluar el clasificador
    y_pred = classifier.predict(X_test)

    print("\nResultados del Clasificador Inicial:")
    print(f"Precisión (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print("Reporte de Clasificación:")
    if len(np.unique(y_test)) > 1:
        print(classification_report(y_test, y_pred, target_names=['No Animal', 'Animal']))
    else:
        print("Solo una clase presente en el conjunto de prueba, no se puede generar un reporte de clasificación completo.")

    # --- Simulación de la Fase de Filtrado ---
    print("\nSimulando la fase de filtrado para YOLOv8...")
    predicted_labels_all_images = classifier.predict(X)

    # Filtrar las imágenes que el clasificador predice como "animal"
    images_for_yolo = [
        filename for i, filename in enumerate(image_filenames_for_classification)
        if predicted_labels_all_images[i] == 1 
    ]

    print(f"De {len(image_filenames_for_classification)} imágenes, {len(images_for_yolo)} imágenes serán pasadas a YOLOv8.")

# Opcional: Guardar la lista de imágenes filtradas para el siguiente paso (se crea siempre)
output_filtered_list_path = os.path.join("ena24", "images_for_yolo.json")
try:
    with open(output_filtered_list_path, 'w') as f:
        json.dump(images_for_yolo, f, indent=4)
    print(f"Lista de imágenes filtradas para YOLOv8 guardada en: {output_filtered_list_path}")
except Exception as e:
    print(f"Error al guardar la lista de imágenes filtradas en JSON: {e}")


print("\n¡El script ha completado todos los pasos hasta la simulación de la fase de filtrado para YOLOv8!")

# ==============================================================================
# --- Paso 5: Detección con YOLOv8 (usando CLAHE) ---
# ==============================================================================

print("\n=============================================")
print("--- Paso 5: Detección con YOLOv8 (usando CLAHE) ---")
print("=============================================")

# Directorio de las imágenes para la inferencia de YOLOv8 (¡AHORA LAS CLAHE!)
input_dir_for_yolo_inference = "ena24/images_ahe_color" 

# Directorio de las imágenes para dibujar las detecciones (también las CLAHE)
input_dir_for_yolo_visualization = "ena24/images_ahe_color" 

# Archivo con la lista de imágenes filtradas por el clasificador inicial
filtered_images_list_path = os.path.join("ena24", "images_for_yolo.json")

# Directorio para guardar las imágenes con detecciones de YOLOv8
output_dir_detections = "ena24/detections"
os.makedirs(output_dir_detections, exist_ok=True)

# Cargar la lista de imágenes filtradas
try:
    with open(filtered_images_list_path, 'r') as f:
        images_to_process_with_yolo = json.load(f)
    print(f"Cargadas {len(images_to_process_with_yolo)} imágenes para procesar con YOLOv8.")
except FileNotFoundError:
    print(f"Error: No se encontró la lista de imágenes filtradas en {filtered_images_list_path}. Esto no debería ocurrir si el paso anterior se ejecutó correctamente. Verificando si el archivo existe pero está vacío o corrupto.")
    # Si el archivo no se encontró, intentamos inicializar la lista como vacía para no detener el script
    images_to_process_with_yolo = [] 
    
if not images_to_process_with_yolo:
    print("No hay imágenes filtradas por el clasificador inicial para pasar a YOLOv8.")
    print("El proceso de detección de YOLOv8 se saltará.")
else:
    # Cargar un modelo YOLOv8 pre-entrenado
    print("Cargando modelo YOLOv8 (yolov8n.pt)...")
    try:
        model = YOLO('yolov8n.pt') 
        print("Modelo YOLOv8 cargado exitosamente.")
    except Exception as e:
        print(f"Error al cargar el modelo YOLOv8: {e}")
        print("Asegúrate de haber instalado 'ultralytics' (pip install ultralytics) y de tener conexión a internet para la descarga inicial del modelo.")
        exit()


    print(f"Iniciando la detección de objetos con YOLOv8 en {len(images_to_process_with_yolo)} imágenes...")

    detection_results = {} 

    for filename in tqdm(images_to_process_with_yolo):
        img_path_for_inference = os.path.join(input_dir_for_yolo_inference, filename)
        img_path_for_visualization = os.path.join(input_dir_for_yolo_visualization, filename)


        if not os.path.exists(img_path_for_inference):
            print(f"Advertencia: La imagen para inferencia {img_path_for_inference} no existe. Saltando.")
            continue
        if not os.path.exists(img_path_for_visualization): 
            print(f"Advertencia: La imagen para visualización {img_path_for_visualization} no existe. Saltando.")
            continue

        try:
            # Realizar la inferencia con YOLOv8 con umbrales más bajos
            results = model(img_path_for_inference, conf=0.1, iou=0.5, verbose=False) 

            # Cargar la imagen CLAHE para dibujar las detecciones
            img_to_draw_on = cv2.imread(img_path_for_visualization)
            if img_to_draw_on is None:
                print(f"Advertencia: No se pudo cargar la imagen CLAHE para dibujar {img_path_for_visualization}. Saltando visualización para esta imagen.")
                continue 


            current_image_detections = []
            for r in results:
                boxes = r.boxes 
                names = r.names 

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = names[class_id]

                    current_image_detections.append({
                        "box": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name
                    })
                    
                    # --- Dibujar el rectángulo y el texto en la imagen CLAHE ---
                    color = (0, 255, 0) # Verde
                    cv2.rectangle(img_to_draw_on, (x1, y1), (x2, y2), color, 2)

                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(img_to_draw_on, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            detection_results[filename] = current_image_detections 


            # Guardar la imagen con las anotaciones en el directorio de detecciones
            output_detection_img_path = os.path.join(output_dir_detections, filename)
            os.makedirs(os.path.dirname(output_detection_img_path), exist_ok=True)
            cv2.imwrite(output_detection_img_path, img_to_draw_on)


        except Exception as e:
            print(f"Error procesando imagen {filename} con YOLOv8: {e}")
            continue

    # Guardar todos los resultados de detección en un archivo JSON
    output_detections_file = os.path.join(output_dir_detections, "yolov8_detections.json")
    try:
        with open(output_detections_file, 'w') as f:
            json.dump(detection_results, f, indent=4)
        print(f"Resultados de detección de YOLOv8 guardados en: {output_detections_file}")
    except Exception as e:
        print(f"Error al guardar los resultados de detección en JSON: {e}")


print("\n¡El script ha completado todos los pasos, incluyendo la detección con YOLOv8!")
print("Puedes revisar las imágenes con detecciones (dibujadas en las versiones CLAHE) en la carpeta 'ena24/detections'.")