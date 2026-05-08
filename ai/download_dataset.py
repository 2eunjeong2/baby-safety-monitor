from roboflow import Roboflow

rf = Roboflow(api_key="66TStgYQ9T9w58A4lvXF")
project = rf.workspace("vtar").project("baby-object-detection-final")
version = project.version(1)
dataset = version.download("yolov8")