from ultralytics import YOLO
import torch

print("CUDA Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

model = YOLO("yolov8n.pt")
model.to("cuda")

imagePath = "processed/frames/real/000/frame_0.jpg"
results = model(imagePath)
for result in results:
    print("Detected classes:", result.boxes.cls)
    print("Confidence scores:", result.boxes.conf)
    print("Bounding boxes:", result.boxes.xyxy)