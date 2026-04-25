import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from videoDataset import VideoDataset
from models.videoModel import create3DModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------- Dataset --------
fullDataset = VideoDataset("dataset_medium", clipLength=16, trainMode=True)
valSize = max(1, int(0.2 * len(fullDataset)))
trainSize = len(fullDataset) - valSize
trainDataset, valDataset = random_split(
    fullDataset,
    [trainSize, valSize],
    generator=torch.Generator().manual_seed(42)
)

trainLoader = DataLoader(trainDataset, batch_size=2, shuffle=True, num_workers=0)

# Use deterministic/clean preprocessing for validation
valSamples = [fullDataset.samples[i] for i in valDataset.indices]
valDatasetEval = VideoDataset(
    "dataset_medium",
    clipLength=16,
    samples=valSamples,
    trainMode=False
)
valLoader = DataLoader(valDatasetEval, batch_size=2, shuffle=False, num_workers=0)


# -------- Model --------
model = create3DModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


# -------- Training --------
epochs = 20
bestValAcc = 0.0
earlyStopPatience = 5
noImproveCount = 0

for epoch in range(epochs):

    model.train()
    totalLoss = 0

    for videos, labels in trainLoader:

        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(videos)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        totalLoss += loss.item()

    # -------- Validation --------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in valLoader:
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    valAcc = (100.0 * correct / total) if total > 0 else 0.0
    scheduler.step()
    print(
        f"Epoch {epoch+1:02d}/{epochs} | "
        f"Train Loss: {totalLoss:.4f} | Val Acc: {valAcc:.2f}% | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    if valAcc > bestValAcc:
        bestValAcc = valAcc
        noImproveCount = 0
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/mvit_model.pth")
        print(f"Saved improved ViT checkpoint (Val Acc: {valAcc:.2f}%)")
    else:
        noImproveCount += 1
        if noImproveCount >= earlyStopPatience:
            print("Early stopping triggered.")
            break

# -------- Save --------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/3d_model_last.pth")

print(f"3D model training completed. Best Val Accuracy: {bestValAcc:.2f}%")