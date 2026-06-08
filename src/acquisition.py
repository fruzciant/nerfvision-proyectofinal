"""
Etapa 1 — Adquisición.
Responsabilidad única: abrir una fuente de video (archivo o webcam) y
entregar frames uno por uno. No sabe nada de detección ni de YOLO.
"""

import cv2


class VideoSource:
    """Envuelve cv2.VideoCapture para iterar frames de archivo o webcam."""

    def __init__(self, source):
        # source: ruta a un archivo de video, o un índice de cámara (0, 1, ...)
        self.source = source
        self.cap = cv2.VideoCapture(source)

    def is_opened(self):
        return self.cap.isOpened()

    @property
    def total_frames(self):
        """Número total de frames (0 o desconocido para webcam)."""
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return max(n, 0)

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS) or 0.0

    def frames(self):
        """Generador que entrega frames BGR hasta que el video termina."""
        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def release(self):
        self.cap.release()

    # Permite usarlo con 'with' para liberar siempre la cámara/archivo
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
