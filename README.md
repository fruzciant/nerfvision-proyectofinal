[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UASsNx--)
# NerfVision — Detección, conteo y estimación de trayectoria de dardos Nerf en video

**Universidad Nacional de Colombia - Sede de La Paz**, asignatura de Visión por Computador.
- **Docente:** Ing. Jose Francisco Ruiz Muñoz PhD.
- **Estudiantes:** Jhoel Ospino, Cristian Cudris, Andrés Amaya.

## Descripción

NerfVision es una aplicación modular de visión por computador que detecta, sigue y cuenta dardos Nerf en video. El sistema implementa un pipeline de cinco etapas sobre un modelo YOLOv11-nano entrenado con transfer learning para dos clases: `dardo` y `pistola`. La pistola se detecta y dibuja, pero no entra en el conteo ni en las trayectorias (solo se cuentan dardos).

La interfaz en Streamlit permite subir un video o usar la cámara en tiempo real, ajustar los parámetros de inferencia y ver el resultado anotado frame a frame.

## Pipeline

El código está organizado en módulos de responsabilidad única dentro de `src/`:

| Etapa | Módulo | Qué hace |
|---|---|---|
| 1 — Adquisición | `src/acquisition.py` | Abre un archivo de video o la webcam con `cv2.VideoCapture` y entrega frames uno por uno |
| 2 — Preprocesamiento | `src/preprocessing.py` | Pasa el frame al detector; opcionalmente aplica realce de contraste (CLAHE) para condiciones de poca luz |
| 3-4 — Extracción / Detección | `src/detection.py` | El backbone CSPDarknet extrae características y la cabeza YOLO produce bounding boxes con NMS; filtra por confianza e IoU |
| 5a — Tracking | `src/detection.py` | ByteTrack asigna IDs persistentes a cada dardo; el conteo acumula los IDs únicos vistos |
| 5b — Visualización | `src/visualization.py` | Dibuja cajas + IDs (vía Ultralytics), traza trayectorias en cian y muestra el contador en el frame |

El orquestador es `app.py`, que conecta todos los módulos y expone los controles al usuario.

## Dataset

Dataset propio construido para el proyecto.

- **Videos crudos:** grabados con celular a 60–240 FPS sobre fondo controlado; incluyen dardos sostenidos, en vuelo y en lanzamiento de izquierda a derecha y viceversa, y la pistola en distintas posiciones.
- **Extracción de frames:** `scripts/extract_frames.py` extrae por tiempo (N frames/segundo real) para que los videos a 240 FPS no generen imágenes redundantes en exceso.
- **Etiquetado:** con Label Studio, clase `dardo` (0) y clase `pistola` (1); criterio inclusivo para frames con motion blur.
- **Split:** `split_dataset.py` organiza el dataset para YOLO. Resultado: **80 imágenes de entrenamiento / 20 de validación**.
- **Data augmentation:** desenfoque gaussiano ligero, variación de brillo ±10 % y transformaciones geométricas sutiles.

```
data/
├── videos_raw/       # videos originales (no versionados)
├── frames_raw/       # frames extraídos (no versionados)
└── dataset/
    ├── images/
    │   ├── train/    # 80 imágenes
    │   └── val/      # 20 imágenes
    └── labels/
        ├── train/
        └── val/
```

## Modelo

- **Arquitectura:** YOLOv11-nano (`yolo11n.pt` como base), afinado con transfer learning.
- **Clases:** `dardo` (0), `pistola` (1).
- **Entrenamiento:** Google Colab con GPU Nvidia T4.
- **Artefacto:** `models/best.pt` (incluido en el repositorio).

## Estructura del repositorio

```
nerfvision-proyectofinal/
├── app.py                  # app Streamlit (orquestador del pipeline)
├── src/
│   ├── acquisition.py      # Etapa 1
│   ├── preprocessing.py    # Etapa 2
│   ├── detection.py        # Etapas 3-4-5a
│   └── visualization.py    # Etapa 5b
├── scripts/
│   └── extract_frames.py   # extrae frames de los videos crudos
├── split_dataset.py        # organiza dataset en train/val para YOLO
├── models/
│   └── best.pt             # modelo entrenado
├── data/
│   └── dataset/            # imágenes y etiquetas listas para YOLO
├── requirements.txt
└── overrides.txt
```

## Instalación y ejecución

**Requisitos:** Python 3.10+, [`uv`](https://github.com/astral-sh/uv).

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd nerfvision-proyectofinal

# Instalar dependencias (el override excluye opencv-python con GUI)
uv pip install -r requirements.txt --overrides overrides.txt

# Correr la aplicación
uv run streamlit run app.py
```

> **Nota sobre OpenCV:** `ultralytics` arrastra `opencv-python` (con GUI) como dependencia transitiva. Este proyecto usa `opencv-python-headless`. El archivo `overrides.txt` excluye la versión con GUI para evitar conflictos en Linux sin librerías gráficas. No desinstalar `opencv-python` manualmente: ambos paquetes comparten el directorio `cv2/` y desinstalar uno borra los archivos del otro.

## Uso de la interfaz

1. Abrir la app en el navegador (por defecto `http://localhost:8501`).
2. Elegir la fuente: **subir un video** (`.mp4`, `.mov`, `.avi`, `.mkv`) o **cámara en tiempo real**.
3. Ajustar en la barra lateral:
   - **Umbral de confianza (conf):** controla qué detecciones se aceptan.
   - **Umbral de NMS (iou):** controla la supresión de cajas solapadas.
   - **Realce para poca luz:** activa CLAHE si las condiciones de iluminación son difíciles.
   - **Mostrar trayectorias:** activa/desactiva el dibujo de recorridos.
4. Pulsar **Procesar**. El frame anotado se muestra en tiempo real junto con el contador de dardos únicos en la barra lateral.

## Dependencias

| Componente | Versión |
|---|---|
| `ultralytics` (YOLOv11 + ByteTrack) | 8.3.0 |
| `opencv-python-headless` | 4.10.0.84 |
| `numpy` | 1.26.4 |
| `streamlit` | 1.40.0 |
| `matplotlib` | 3.9.2 |
| `lap` (asignación para tracking) | 0.5.13 |

El entrenamiento se realizó en Colab; la app corre en CPU local sin hardware adicional.

## Limitaciones conocidas

- **Motion blur:** dardos a velocidad máxima pueden aparecer muy borrosos. Se mitiga con captura a alto FPS y etiquetado inclusivo, pero puede afectar el recall.
- **Conteo doble:** si un dardo desaparece varios frames (por blur o salida del encuadre) y reaparece, ByteTrack puede asignarle un ID nuevo. Se mitiga con alto FPS y entorno controlado.
