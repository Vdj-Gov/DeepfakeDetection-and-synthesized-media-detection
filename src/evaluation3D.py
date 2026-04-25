import os
import cv2
import torch
import torchvision.models.video as models
import torchvision.transforms as transforms
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- Load model --------
model = models.r3d_18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("models/3d_model.pth", map_location=device))
model.to(device)
model.eval()


from torch.utils.data import DataLoader
from videoDataset import VideoDataset
from retinaface import RetinaFace

clipLength = 16
root = "dataset_subset"

print("Warming up RetinaFace (first call can be slow)...")
try:
    _ = RetinaFace.detect_faces(np.zeros((224, 224, 3), dtype=np.uint8))
    print("RetinaFace warmup done.")
except Exception as e:
    print(f"RetinaFace warmup warning: {e}")

print("Loading VideoDataset (this incorporates Face Extraction via RetinaFace)...")
dataset = VideoDataset(root, clipLength=clipLength)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
print(f"Total videos for 3D evaluation: {len(dataset)}")

correct = 0
total = 0

for idx, (videos, labels) in enumerate(loader, start=1):
    videos = videos.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        output = model(videos)
        pred = torch.argmax(output, dim=1)

    correct += (pred == labels).sum().item()
    total += labels.size(0)
    if idx % 10 == 0 or idx == len(loader):
        runningAcc = (correct / total) * 100
        print(f"Processed {idx}/{len(loader)} videos - Running Accuracy: {runningAcc:.2f}%")

accuracy = correct / total * 100
print(f"3D Model Accuracy: {accuracy:.2f}%")