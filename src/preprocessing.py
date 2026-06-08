"""
Etapa 2 — Preprocesamiento.
Responsabilidad única: dejar el frame listo para el detector.

Nota honesta: Ultralytics hace internamente su propio preprocesamiento
(resize a imgsz, normalización, BGR→RGB) cuando le pasas un frame. Este
módulo existe para EVIDENCIAR la etapa en el pipeline y para poder aplicar
ajustes propios si se quisieran (p. ej. realce de contraste en el aula
con poca luz). Por defecto es un paso ligero y transparente.
"""

import cv2


def to_rgb(frame_bgr):
    """OpenCV carga en BGR; varias visualizaciones esperan RGB."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def enhance_low_light(frame_bgr, clip_limit=2.0, tile=8):
    """
    Realce opcional de contraste para frames con poca luz (aula).
    Aplica CLAHE sobre el canal de luminancia. Útil como experimento;
    NO está activo por defecto en el pipeline (el detector ya es robusto).
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def preprocess(frame_bgr, low_light=False):
    """
    Punto de entrada de la etapa. Devuelve el frame listo para el detector.
    Por defecto NO modifica el frame (Ultralytics se encarga del resize y
    la normalización). Si low_light=True, aplica realce de contraste.
    """
    if low_light:
        return enhance_low_light(frame_bgr)
    return frame_bgr
