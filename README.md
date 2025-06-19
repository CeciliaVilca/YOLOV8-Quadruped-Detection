Este proyecto tiene como objetivo replicar la metodología propuesta en el artículo "Novel Animal Detection System: Cascaded YOLOv8 With Adaptive Preprocessing and Feature Extraction" para la detección de animales salvajes. A continuación, se detallan los pasos implementados hasta la fecha.

1. Configuración Inicial del Entorno y Descarga de Datos

    Descripción: Preparación del entorno de trabajo (Visual Code Python) y gestión de los datos iniciales.
        Instalación de librerías esenciales en Python (Ultralytics, OpenCV, Scikit-image, Scikit-learn, TensorFlow).
        Definición de directorios para la organización de los datos de entrada y salida.
        Descarga y filtrado de metadatos del dataset ENA24 (o similar a COCO) para identificar imágenes que contienen "cuadrúpedos".
        Descarga de un subconjunto de imágenes filtradas (ej. 100 imágenes) para las fases iniciales de prueba y desarrollo.

2. Preprocesamiento Adaptativo: Ecualización de Histograma Adaptativa (AHE)

    Descripción: Mejora del contraste de las imágenes para optimizar las etapas posteriores de segmentación y extracción de características.
        Implementación de CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Aplicación de CLAHE al canal de luminancia (L*) de las imágenes a color (BGR -> LAB -> L* con CLAHE -> LAB -> BGR), preservando la información cromática original.
        Guardado de las imágenes preprocesadas en un directorio específico (ena24/images_ahe_color).

3. Segmentación Basada en Superpíxeles

    Este es un paso de preprocesamiento avanzado que simplifica las imágenes para el análisis de características.

   Segmentación Semántica: Simula una "segmentación rápida basada en superpíxeles" mediante la combinación de dos técnicas:
            SLIC (Simple Linear Iterative Clustering): Agrupa píxeles similares en "superpíxeles", que son regiones coherentes y perceptualmente significativas. Esto reduce el número de unidades de procesamiento de píxeles individuales a grupos más grandes.
            K-Means: Aplica el algoritmo K-Means a los colores promedio de estos superpíxeles para agruparlos en un número predefinido de clústeres de color. Esto ayuda a simplificar la imagen y destacar regiones de interés.
            Paralelización (¡Optimización Añadida!): Aquí es donde se ha implementado la aceleración con multiprocessing. En lugar de procesar las imágenes una por una, el script divide la carga entre múltiples núcleos de la CPU, permitiendo que varias imágenes se segmenten simultáneamente, lo que reduce drásticamente el tiempo total de este paso.
   Las imágenes segmentadas se guardan en ena24/images_segmented_kmeans.

     <p align="center">
    <img src="evidencias/Plan1.png" alt="Mostrando algunas imágenes para comparación (Original, CLAHE, Segmentada)" width="700"/>
    </p>

4. Extracción de Características
   Este paso crucial prepara los datos para la clasificación.

   De las imágenes segmentadas (imágenes preprocesadas), el script extrae vectores numéricos que representan sus características visuales. Utiliza tres tipos de descriptores:
        LBP (Local Binary Pattern): Un descriptor de textura que captura patrones locales en la imagen.
        ResNet50: Utiliza ResNet50 para extraer características de alto nivel que son muy efectivas para el reconocimiento de objetos.
        MobileNet (como aproximación a Darknet19): Usa MobileNet para obtener características ligeras y eficientes, similar a lo que haría Darknet19.
    Combinación de Características: Los vectores de características de LBP, ResNet50 y MobileNet se combinan en un único vector para cada imagen.
    Almacenamiento: Estas características se guardan en un archivo JSON (ena24/features/image_features.json) para que no sea necesario volver a extraerlas en ejecuciones futuras.

5. Clasificación Inicial y Filtrado
                                                               Aplicación de YOLOv8 para la detección de animales cuadrúpedos en imágenes naturales

Este paso actúa como un "pre-filtro" para optimizar el siguiente paso de detección.

Entrenamiento del Clasificador: Entrena un modelo de Random Forest Classifier utilizando las características combinadas extraídas en el paso anterior. El objetivo de este clasificador es predecir si una imagen contiene un "animal" (etiquetado como 1) o "no animal". Aunque en este script todas las imágenes descargadas inicialmente se etiquetan como "animal" para este clasificador, en un flujo real este paso podría usarse para descartar imágenes irrelevantes (por ejemplo, sin animales).

Simulación de Filtrado: Después de entrenar el clasificador, se predice la clase para todas las imágenes. Solo aquellas imágenes que el clasificador predice como "animal" son seleccionadas para el siguiente paso, la detección con YOLOv8.

 Salida: Se genera un archivo ena24/images_for_yolo.json con los nombres de las imágenes filtradas.

 6. Detección con YOLOv8 en Cascada

Este es el paso final para identificar y localizar objetos específicos dentro de las imágenes.

Modelos YOLOv8: Carga dos modelos YOLOv8 pre-entrenados:
        yolov8n.pt (nano): Un modelo más pequeño y rápido para una detección inicial "gruesa".
        yolov8s.pt (small): Un modelo ligeramente más grande y preciso para un refinamiento.

Proceso en Cascada:
        Detección Gruesa: El modelo yolov8n.pt se ejecuta primero sobre la imagen completa para identificar posibles "Regiones de Interés" (ROIs) donde podría haber un objeto.
        Refinamiento: Luego, cada una de esas ROIs detectadas por el primer modelo es recortada y pasada al modelo yolov8s.pt para una detección más precisa y detallada dentro de esas regiones. Esto ayuda a reducir falsos positivos y mejorar la localización.
    Visualización y Guardado: Las detecciones finales (bounding boxes, confianza y nombre de la clase) se dibujan sobre las imágenes originales (las imágenes CLAHE) y se guardan en ena24/cascaded_detections.
    Resultados JSON: Un archivo yolov8_cascaded_detections.json almacena los detalles de todas las detecciones para análisis posteriores.
   
   
    
