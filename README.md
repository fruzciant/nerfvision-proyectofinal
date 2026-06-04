[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UASsNx--)
# Propuesta de Proyecto Final para Visión por Computador
**Universidad Nacional de Colombia - Sede de La Paz**, asignatura de Visión por Computador.
- **Docente:** Ing. Jose Francisco Ruiz Muñoz PhD.
- **Estudiantes:** Jhoel Ospino, Cristian Cudris, Andrés Amaya.
- **Fecha:** 4 de junio de 2026

# Proyecto
## NerfVision: Detección, conteo y estimación de trayectoria de dardos Nerf en video con YOLOv8 en un entorno controlado.

## Descripción
Este proyecto aborda detección automática de dardos Nerf en secuencias de video, un problema relevante para detectar objetos pequeños, de alto contraste y en movimiento rápido, con un desafío técnico en la parte del movimiento (motion blur), por la velocidad del proyectil.

El sistema recibirá video (archivo pregrabado o cámara en tiempo real [menos eficiente]), detectando cada dardo presente en lo frames, siguiéndolo a lo largo de la secuencia para contar cuántos dardos distintos aparecen y reconstruir su trayectoria sobre la imagen. El contexto de uso es un entorno controlado (fondo liso, con una iluminación estable), permitiendo abordar el problema con un dataset pequeño y un modelo ligero sin sacrificar el rigor del pipeline.

Esta aplicación será un prototipo académico (MVP) para mostrar que se dominan las cinco etapas del pipeline de la asignatura, no está pensado para ser un producto de grado industrial (solo aprendizaje apenas). El alcance está acotado a una sola clase de objeto en condiciones controladas para garantizar viabilidad y reproducibilidad.

## Objetivos
### Objetivo general
Desarrollar una aplicación modular de visión por computador, con interfaz de usuario, que detecte, cuente y estime la trayectoria de dardos Nerf en video sobre un entorno controlado, aplicando integralmente las cinco etapas del pipeline definids en las instrucciones del proyecto.

### Objetivos específicos
1. Construir un dataset de dardos Nerf con video capturado en una alta tasa de cuadros (grabado a 120 o a 240 FPS con la cámara de un iPhone y luego acelarando el vídeo), con criterio de etiquetado inclusivo para frames con desenfoque por movimiento.
2. Entrenar un detector de objetos YOLOv8-nano mediante *transfer learning* para la clase única `dardo`, la elección del modelo es por su eficiencia y capacidad de generalización con pocos datos (por el tiempo).
3. Implementar un pipeline modular de cinco etapas (adquisición, preprocesamiento, detección, seguimiento/conteo, postprocesamiento y visualización) con responsabilidad única por módulo.
4. Integrar un componente de seguimiento (*tracking*) que asigne identidades persistentes a cada dardo para permitir el conteo de instancias únicas y la reconstrucción de su trayectoria.
5. Desarrollar una interfaz en Streamlit que permita cargar video o usar la cámara, ejecutar el pipeline y visualizar los resultados anotados de cada etapa.
6. Evaluar cuantitativamente el desempeño del sistema con métricas apropiadas (mAP50, F1-score, FPS) y analizar errores observados, particularmente los que se asocian con *motion blur* y al seguimiento.

## Descripción del pipeline
Estas serán las etapas que resolveremos con su entrada, salida y técnica empleada.
### Etapa 1 - Adquisición
+ **Entrada:** escena física con uno o varios dardos Nerf (sostenidos, en vuelo o estáticos) sobre fondo controlado.
+ **Salida:** secuencia de frames RGB (video pregrabado `.mp4` o flujo de webcam en tiempo real [yo creo que este puede ser un poco menos confiable]).
+ **Técnica:** captura con cámara de dispositivo móvil a 60, 120 o 240 FPS (en cámara lenta también) para reducir el desplazamiento del proyectil por frame y obtener contornos más definidos. El sistema soporta dos modos de entrada: archivo de video y cámara en tiempo real, leídos con OpenCV (`cv2.VideoCapture`).

    **Fundamentación técnica:** entre mayor es la tasa de cuadros, menor es el desplazamiento del objeto entre frames consecutivos. Un dardo a ~18m/s recorre ~60cm entre frames a 30FPS, pero solo ~15cm a 120 FPS. Esa reducción del desplazamiento por frame, junto con la exposición más corta que suele forzar el modo cámara lenta, hace menor el motion blur y mejora la detectabilidad del objeto en vuelo.

### Etapa 2 - Preprocesamiento
+ **Entrada:** frame RGB crudo de la etapa de adquisición.
+ **Salida:** frame normalizado y redimensionado, listo para el detector.
+ **Técnica:** redimensionamiento a 640x640px (resolución nativa d eentrada de YOLOv8), normalización de intensidades al rango $[0,1]$ y conversión del espacio de color BGRtoRGB (porque OpenCV carga en BGR; YOLO espera RGB). Estas operaciones se delegan en buena parte al preprocesador interno de Ultralytics, pero se documentan y exponen como módulo propio para evidenciar la etapa.

### Etapa 3 - Segmentación
+ **Entrada:** frame preprocesado.
+ **Salida:** mapas de características y regiones candidatas (propuestas de bounding box con score).
+ **Técnica:** extracción jerárquica de características mediante la red convolucional troncal (backbone CSPDarknet) de YOLOv8. Las primeras capas detectan bordes, contornos y colores; las profundas combinan esos patrones en representaciones del objeto completo. Esta etapa es intrínseca a la arquitectura del detector y se documenta visualizando los mapas de activación intermedios sobre frames de ejemplo.

### Etapa 4 - Reconocimiento / detección
+ **Entrada:** características extraídas por el backbone.
+ **Salida:*** lista de detecciones por frame, cada una con bounding box $(x_{min}, y_{min}, x_{max}, y_{max})$, clase `dardo` y score de confianza.
+ **Técnica:** detector **YOLOv8-nano** afinado por transfer learning sobre la clase única `dardo`. La cabeza de detección predice múltiples cajas simultáneas y aplica *Non-Maximum Suppression* (NMS) para descartar solapamientos redundantes. Los parámetros de inferencia `conf` (umbral de confianza) e `iou` (umbral de NMS) se exponen como ajustables.

### Etapa 5 - Postprocesamiento y visualización
+ **Entrada:** detecciones por frame.
+ **Salida:** video/imagen anotado con cajas, identidades, conteo acumulado y trayectorias, más un reporte de métricas.
+ **Técnica**
    + **Seguimiento (tracking):** se aplica **ByteTrack** (integrado en Ultralytics con `model.track(...)`) para asignar un ID persistente a cada dardo a lo largo de los frames.
    + **Conteo:** se cuenta el número de IDs únicos para reportar cuántos dardos distintos aparecieron en la secuencia.
    + **Trayectoria:** se almacena el centroide de cada ID por frame y se traza la polilínea de su recorrido sobre el vídeo.
    + **Visualización:** anotación con OpenCV (cajas, etiqueta, ID, score) y superposición de la trayectoria; render en la interfaz en Streamlit.



