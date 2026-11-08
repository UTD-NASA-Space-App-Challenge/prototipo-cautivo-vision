from roboflow import Roboflow

rf = Roboflow(api_key="4cNqJPip6STbgUSNy5oL")
project = rf.workspace("said-uyao1").project("rock-bvssc")
version = project.version(1)
dataset = version.download("yolov8")
