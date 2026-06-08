"""
NerfVision — App Streamlit
Nivel 0-1: subir video + detección frame a frame.

Estado actual:
  - Usa el modelo GENÉRICO yolo11n.pt (COCO) como sustituto temporal.
    Detectará personas, sillas, etc. — no dardos todavía. Es a propósito:
    sirve para construir y probar la interfaz ANTES de tener best.pt.
  - Cuando tengas tu modelo entrenado, cambia MODEL_PATH a "models/best.pt".

Próximos niveles (ver comentarios marcados con  # >>> NIVEL N):
  - Nivel 2: tracking + conteo de IDs únicos
  - Nivel 3: trayectorias
  - Nivel 4: sliders conf/iou + modo webcam

Correr:
    streamlit run app.py
"""

import tempfile

import cv2
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
# Nivel 1: modelo genérico para construir la UI.
# Nivel 4: reemplazar por "models/best.pt" cuando esté entrenado.
MODEL_PATH = "yolo11n.pt"


@st.cache_resource
def load_model(path):
    """Carga el modelo una sola vez y lo cachea entre reruns de Streamlit."""
    return YOLO(path)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.set_page_config(page_title="NerfVision", layout="wide")
st.title("NerfVision — Detección de dardos Nerf")
st.caption(
    "Nivel 0-1: detección frame a frame. "
    "Usando modelo genérico (COCO) hasta integrar best.pt."
)

# >>> NIVEL 4: aquí irán los sliders de conf / iou en la barra lateral.
# conf = st.sidebar.slider("Confianza (conf)", 0.0, 1.0, 0.25, 0.05)
# iou  = st.sidebar.slider("NMS (iou)", 0.0, 1.0, 0.7, 0.05)

# >>> NIVEL 4: aquí irá el selector de modo (subir video / webcam).
# mode = st.radio("Fuente", ["Subir video", "Cámara en tiempo real"])

uploaded = st.file_uploader(
    "Sube un video", type=["mp4", "mov", "avi", "mkv"]
)

if uploaded is not None:
    # Streamlit entrega el archivo en memoria; OpenCV necesita una ruta,
    # así que lo escribimos a un archivo temporal.
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.flush()

    model = load_model(MODEL_PATH)

    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("No se pudo abrir el video. Prueba otro formato.")
        st.stop()

    frame_slot = st.empty()        # contenedor que se actualiza por frame
    progress = st.progress(0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # >>> NIVEL 2: aquí inicializarías el tracker / set de IDs vistos.
    # seen_ids = set()
    # >>> NIVEL 3: aquí el diccionario de trayectorias.
    # trajectories = {}

    idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # --- Inferencia (Nivel 1: detección simple) ---
        results = model(frame, verbose=False)
        annotated = results[0].plot()   # dibuja las cajas sobre el frame

        # >>> NIVEL 2: cambiar la línea de arriba por:
        #     results = model.track(frame, persist=True,
        #                            tracker="bytetrack.yaml", verbose=False)
        #     y recorrer results[0].boxes.id para contar IDs únicos.

        # >>> NIVEL 3: calcular el centroide de cada caja y acumularlo en
        #     trajectories[id], luego dibujar la polilínea sobre `annotated`.

        # Streamlit muestra en RGB; OpenCV trae BGR.
        frame_slot.image(annotated, channels="BGR", use_container_width=True)

        idx += 1
        progress.progress(min(idx / total, 1.0))

    cap.release()
    st.success("Procesamiento terminado.")
else:
    st.info("Sube un video para empezar.")
