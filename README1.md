# NerfVision

Detección, conteo y estimación de trayectoria de dardos Nerf en video, mediante YOLO11n + ByteTrack + Streamlit.

Proyecto Final — Visión por Computador
Universidad Nacional de Colombia, Sede de La Paz

---

## Estado actual

App en **Nivel 0-1**: sube un video y corre detección frame a frame.
Por ahora usa el modelo genérico `yolo11n.pt` (COCO) como sustituto temporal
mientras se entrena el detector de dardos. Detectará objetos genéricos
(personas, sillas…), no dardos todavía — es a propósito, para construir y
probar la interfaz antes de tener el modelo entrenado.

## Estructura

```
nerfvision/
├── app.py              # interfaz Streamlit (Nivel 0-1)
├── requirements.txt
├── .gitignore
├── src/                # módulos del pipeline (se llenan en niveles 2-4)
├── scripts/            # extract_frames.py, split_dataset.py
├── notebooks/          # notebook de entrenamiento en Colab
├── docs/               # diagrama del pipeline
├── models/             # best.pt entrenado (cuando exista)
└── data/               # dataset
```

## Instalación y arranque (PC local — Fedora / openSUSE)

```bash
# 1. Clonar y entrar
git clone https://github.com/fruzciant/nerfvision-proyectofinal.git
cd nerfvision-proyectofinal

# 2. Entorno virtual de la APP
python3 -m venv .venv
source .venv/bin/activate

# 3. Dependencias de la app
pip install -r requirements.txt

# 4. Correr
streamlit run app.py
```

La primera vez, Ultralytics descarga `yolo11n.pt` automáticamente
(necesita conexión). Después funciona offline.

> **Label Studio va en un entorno APARTE.** Es pesado y conviene no mezclarlo
> con la app:
> ```bash
> python3 -m venv .venv-labelstudio
> source .venv-labelstudio/bin/activate
> pip install label-studio==1.13.1
> label-studio start
> ```

## Estrategia de desarrollo incremental

La app crece por niveles; cada uno corre y se prueba antes de seguir:

- **Nivel 0** — Streamlit arranca, sube y muestra video. ✅
- **Nivel 1** — Detección frame a frame con modelo genérico. ✅ (estado actual)
- **Nivel 2** — Tracking con ByteTrack (`model.track(persist=True)`) + conteo de IDs únicos.
- **Nivel 3** — Trayectorias (polilínea de centroides por ID).
- **Nivel 4** — Integrar `models/best.pt` + sliders conf/iou + modo webcam.

Los puntos de extensión están marcados en `app.py` con `# >>> NIVEL N`.

Para pasar al detector de dardos real: entrenar (ver `notebooks/`),
colocar el modelo en `models/best.pt` y cambiar `MODEL_PATH` en `app.py`.

## Pipeline (5 etapas)

1. **Adquisición** — video (archivo o webcam) vía OpenCV.
2. **Preprocesamiento** — resize 640, normalización, BGR→RGB.
3. **Extracción de características** — backbone CSPDarknet de YOLO11.
4. **Detección** — cabeza YOLO11n + NMS → cajas de dardos.
5. **Postprocesamiento** — ByteTrack (IDs), conteo, trayectorias, visualización.
