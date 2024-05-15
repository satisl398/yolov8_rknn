from ultralytics import YOLO

imgsz = 640
model = YOLO(r'yourpath\yolov8n-obb.pt')

model.export(format='rknn', imgsz=imgsz, opset=19, simplify=True)
