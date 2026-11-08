import cv2
import time
from ultralytics import YOLO

# 1. Cargar el modelo base (best.pt)
model = YOLO('runs/detect/train-4/weights/best.pt')

# 2. Inicializar webcam Logitech con codec MJPG para máxima velocidad
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("[ERROR] No se pudo acceder a la webcam Logitech.")
    exit()

cv2.namedWindow("Logitech Vision System - NASA Rover", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Logitech Vision System - NASA Rover", 1280, 720)

print("[INFO] Sistema de visión activo. Presiona 'q' para salir.")
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] No se recibe señal de video.")
        break

    # Inferencia en GPU Orin Nano sin lanzar warning messages
    results = model.predict(
        source=frame,
        conf=0.5,
        device=0,
        verbose=False
    )

    # Renderizar cajas delimitadoras (bounding boxes)
    annotated_frame = results[0].plot()

    # Calcular FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # Telemetría en pantalla
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, "NASA Space Apps - UTD Rover", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Logitech Vision System - NASA Rover", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
