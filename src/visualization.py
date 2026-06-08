"""
Etapa 5b — Visualización.
Responsabilidad única: dibujar sobre el frame las cajas + IDs (vía Ultralytics)
y superponer las trayectorias acumuladas. No decide nada de detección.
"""

import cv2


def draw_overlay(results, trajectories, draw_trajectories=True):
    """
    Devuelve un frame anotado (BGR).

    - results: objeto Results de Ultralytics; results.plot() ya dibuja
      las cajas, la clase, el ID de track y el score.
    - trajectories: dict id -> lista de centroides, para trazar el recorrido.
    """
    # Ultralytics dibuja cajas + etiquetas + IDs de tracking automáticamente
    annotated = results.plot()

    if draw_trajectories:
        for points in trajectories.values():
            # Necesitamos al menos 2 puntos para trazar una línea
            for i in range(1, len(points)):
                p1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                p2 = (int(points[i][0]), int(points[i][1]))
                cv2.line(annotated, p1, p2, (0, 255, 255), 2)
            # Punto actual (cabeza de la trayectoria)
            if points:
                last = (int(points[-1][0]), int(points[-1][1]))
                cv2.circle(annotated, last, 4, (0, 255, 255), -1)

    return annotated


def draw_count(frame, count):
    """Dibuja el contador de dardos únicos en una esquina del frame."""
    text = f"Dardos: {count}"
    cv2.putText(
        frame, text, (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA,   # borde
    )
    cv2.putText(
        frame, text, (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA,  # relleno
    )
    return frame
