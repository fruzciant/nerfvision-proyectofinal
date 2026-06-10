"""
Etapas 3-4 — Extracción de características + Detección — y Etapa 5a — Tracking.

En YOLO, la extracción de características (backbone CSPDarknet) y la detección
(cabeza + NMS) viven dentro de la MISMA red, así que se cargan y ejecutan
juntas con un solo modelo. El tracking (ByteTrack) se ejecuta sobre las
detecciones para asignar IDs persistentes y permitir el conteo.

El modelo distingue DOS clases: 'dardo' (0) y 'pistola' (1). El conteo y las
trayectorias se calculan SOLO para los dardos: contar pistolas no tiene
sentido (suele haber una fija en mano). Las pistolas igual se detectan y se
dibujan, pero no entran en el contador ni en las trayectorias.

Este módulo encapsula el modelo y el estado del tracker.
"""

from ultralytics import YOLO

# Índice de la clase a contar/seguir. Debe coincidir con el orden en data.yaml:
#   names: ['dardo', 'pistola']  ->  dardo = 0
# Con el modelo genérico (yolo11n.pt de COCO) NO existe la clase 'dardo';
# por eso DART_CLASS_ID solo aplica al modelo entrenado. Ver count_only_class.
DART_CLASS_ID = 0


class DartDetector:
    """
    Detecta dardos y pistolas, y sigue los DARDOS a lo largo del video.

    - Etapas 3-4: el modelo YOLO extrae características y produce detecciones.
    - Etapa 5a: model.track(persist=True) usa ByteTrack para mantener IDs.
      El conteo acumula los IDs únicos vistos, pero solo de la clase dardo.
    """

    def __init__(self, model_path="yolo11n.pt", count_only_class=DART_CLASS_ID):
        # Nivel 1-3 (pruebas): yolo11n.pt genérico (COCO).
        # Nivel 4 (final): "models/best.pt" entrenado en dardo + pistola.
        self.model = YOLO(model_path)
        # Clase que se cuenta y se sigue. None = contar todas las clases
        # (útil al probar con el modelo genérico, que no tiene 'dardo').
        self.count_only_class = count_only_class
        self.seen_ids = set()        # IDs únicos de dardo -> conteo
        self.trajectories = {}       # id -> [(cx, cy), ...] centroides por frame

    def reset(self):
        """Limpia el estado entre videos distintos."""
        self.seen_ids.clear()
        self.trajectories.clear()

    def detect_and_track(self, frame, conf=0.25, iou=0.7):
        """
        Procesa un frame: detección + tracking.
        Devuelve el objeto Results de Ultralytics (para que la etapa de
        visualización lo dibuje) y actualiza conteo y trayectorias de dardos.
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
            clss = r.boxes.cls.cpu().numpy().astype(int)   # clase de cada caja
            for box_id, box, cls_id in zip(ids, xyxy, clss):
                # Filtro por clase: si count_only_class está fijado, solo
                # contamos/seguimos esa clase (los dardos). La pistola se
                # ignora aquí, pero se dibuja igual en visualización.
                if self.count_only_class is not None and cls_id != self.count_only_class:
                    continue
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
