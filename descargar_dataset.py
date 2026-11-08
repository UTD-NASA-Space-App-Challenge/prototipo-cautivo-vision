from roboflow import Roboflow

# Conectar a la API pública de Roboflow
rf = Roboflow(api_key="4cNqJPip6STbgUSNy5oL") # O la clave de tu cuenta de Roboflow

# Descargar dataset etiquetado en formato YOLOv8
project = rf.workspace("nasa-rocks").project("lunar-and-terrestrial-rocks")
dataset = project.version(1).download("yolov8", location="/home/jetson/Desktop/UTD/NASA/rock-dataset")

print("[INFO] Dataset successfully downloaded!")
