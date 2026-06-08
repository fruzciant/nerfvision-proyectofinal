"""
Etapas 3-4 — Extracción de características + Detección — y Etapa 5a — Tracking.

En YOLO, la extracción de características (backbone CSPDarknet) y la detección
(cabeza + NMS) viven dentro de la MISMA red, así que se cargan y ejecutan
juntas con un solo modelo. El tracking (ByteTrack) se ejecuta sobre las
detecciones para asignar IDs persistentes y permitir el conteo.

Este módulo encapsula el modelo y el estado del tracker.
"""

from ultralytics import YOLO


class DartDetector:
    """
    Detecta y sigue dardos a lo largo de los frames de un video.

    - Etapas 3-4: el modelo YOLO extrae características y produce detecciones.
    - Etapa 5a: model.track(persist=True) usa ByteTrack para mantener IDs.
      El conteo se hace acumulando los IDs únicos vistos en un set.
    """

    def __init__(self, model_path="yolo11n.pt"):
        # Nivel 1-3 (pruebas): yolo11n.pt genérico (COCO).
        # Nivel 4 (final): "models/best.pt" entrenado en dardos.
        self.model = YOLO(model_path)
        self.seen_ids = set()        # IDs únicos -> conteo
        self.trajectories = {}       # id -> [(cx, cy), ...] centroides por frame

    def reset(self):
        """Limpia el estado entre videos distintos."""
        self.seen_ids.clear()
        self.trajectories.clear()

    def detect_and_track(self, frame, conf=0.25, iou=0.7):
        """
        Procesa un frame: detección + tracking.
        Devuelve el objeto Results de Ultralytics (para que la etapa de
        visualización lo dibuje) y actualiza conteo y trayectorias.
        """
        results = self.model.track(
            frame,
            persist=True,             # continúa el tracking entre frames
            tracker="bytetrack.yaml",
            conf=conf,
            iou=iou,
            verbose=False,
        )
        r = results[0]

        # Si hay IDs asignados, actualizamos conteo y trayectorias
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.cpu().numpy()
            for box_id, box in zip(ids, xyxy):
                bid = int(box_id)
                self.seen_ids.add(bid)
                cx = float((box[0] + box[2]) / 2)
                cy = float((box[1] + box[3]) / 2)
                self.trajectories.setdefault(bid, []).append((cx, cy))

        return r

    @property
    def count(self):
        """Número de dardos únicos detectados hasta el momento."""
        return len(self.seen_ids)
