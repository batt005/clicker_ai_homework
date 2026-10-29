from ultralytics import YOLO

model = YOLO("runs/detect/cookie_detector-4/weights/best.pt")  # поправьте на ваш путь
model.export(format="onnx", opset=12, simplify=True)