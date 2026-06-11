"""
NerfVision — App Streamlit (modular)
Orquesta las 5 etapas del pipeline, cada una en su módulo en src/.

Estado: usa el modelo genérico yolo11n.pt (COCO) como sustituto temporal.
Para el detector de dardos real: entrena (ver notebooks/), coloca el modelo
en models/best.pt y cambia MODEL_PATH abajo.

Correr:
    uv run streamlit run app.py
"""

import tempfile

import streamlit as st

from src.acquisition import VideoSource
from src.preprocessing import preprocess
from src.detection import DartDetector
from src.visualization import draw_overlay, draw_count

# Nivel 1-3 (pruebas): "yolo11n.pt"  |  Nivel 4 (final): "models/best.pt"
MODEL_PATH = "models/best.pt"

# ¿Es el modelo entrenado (dardo+pistola) o el genérico de COCO?
# Con el genérico NO existe la clase 'dardo', así que contamos TODO (None).
# Con best.pt activamos el filtro para contar SOLO dardos (clase 0).
USING_TRAINED_MODEL = MODEL_PATH.endswith("best.pt")
COUNT_CLASS = 0 if USING_TRAINED_MODEL else None   # 0 = dardo en data.yaml


@st.cache_resource
def load_detector(path, count_class):
    """Carga el detector una sola vez y lo cachea entre reruns."""
    return DartDetector(path, count_only_class=count_class)


st.set_page_config(page_title="NerfVision", layout="wide")
st.title("NerfVision — Detección y conteo de dardos Nerf")
st.caption(
    "Pipeline modular de 5 etapas. "
    "Usando modelo genérico (COCO) hasta integrar best.pt."
)

# --- Controles (Etapa 4: parámetros de inferencia) ---
conf = st.sidebar.slider("Umbral de confianza (conf)", 0.0, 1.0, 0.25, 0.05)
iou = st.sidebar.slider("Umbral de NMS (iou)", 0.0, 1.0, 0.7, 0.05)
low_light = st.sidebar.checkbox("Realce para poca luz (aula)", value=False)
show_traj = st.sidebar.checkbox("Mostrar trayectorias", value=True)

mode = st.radio("Fuente de entrada", ["Subir video", "Cámara en tiempo real"])

uploaded = None
if mode == "Subir video":
    uploaded = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"])

start = st.button("Procesar")

if start:
    detector = load_detector(MODEL_PATH, COUNT_CLASS)
    detector.reset()   # limpia estado de un video anterior

    # --- Etapa 1: adquisición ---
    if mode == "Subir video":
        if uploaded is None:
            st.warning("Sube un video primero.")
            st.stop()
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()
        source = tfile.name
    else:
        source = 0   # webcam por defecto

    frame_slot = st.empty()
    count_slot = st.sidebar.empty()
    progress = st.progress(0.0)

    with VideoSource(source) as video:
        if not video.is_opened():
            st.error("No se pudo abrir la fuente de video.")
            st.stop()
        total = video.total_frames or 1

        for idx, frame in enumerate(video.frames()):
            # --- Etapa 2: preprocesamiento ---
            frame = preprocess(frame, low_light=low_light)

            # --- Etapas 3-4-5a: detección + tracking ---
            results = detector.detect_and_track(frame, conf=conf, iou=iou)

            # --- Etapa 5b: visualización ---
            annotated = draw_overlay(results, detector.trajectories,
                                     draw_trajectories=show_traj)
            annotated = draw_count(annotated, detector.count)

            frame_slot.image(annotated, channels="BGR",
                             use_container_width=True)
            count_slot.metric("Dardos únicos", detector.count)
            progress.progress(min((idx + 1) / total, 1.0))

    st.success(f"Listo. Dardos únicos detectados: {detector.count}")
