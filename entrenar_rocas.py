from ultralytics import YOLO

model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    results = model.train(
        data='rock-1/data.yaml',
        epochs=50,
        imgsz=640,
        batch=4,         # Reduce el consumo en la GPU
        workers=1,       # CRÍTICO: Limita los hilos de CPU para no desbordar la RAM
        device=0
    )
