import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import random
import shutil

DATASET_REAL = "dataset_subset/real"
DATASET_FAKE = "dataset_subset/fake"
SPLIT_DIR    = "processed/splits"
MODEL_PATH   = "deepfake_detector.pth"
EPOCHS       = 15
BATCH_SIZE   = 32
LR           = 0.00005
IMG_SIZE     = 224
TRAIN_RATIO  = 0.8
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def splitDataset():
    for cls, src in [("real", DATASET_REAL), ("fake", DATASET_FAKE)]:
        images = [f for f in os.listdir(src) if f.endswith((".jpg", ".png"))]
        random.shuffle(images)
        split = int(len(images) * TRAIN_RATIO)
        for phase, files in [("train", images[:split]), ("val", images[split:])]:
            dest = os.path.join(SPLIT_DIR, phase, cls)
            os.makedirs(dest, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(src, f), os.path.join(dest, f))
    print("Dataset split done.")

class FaceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
        for label, cls in enumerate(["real", "fake"]):
            folder = os.path.join(root, cls)
            for f in os.listdir(folder):
                if f.endswith((".jpg", ".png")):
                    self.samples.append((os.path.join(folder, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

trainTransform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
valTransform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Clear old splits and redo
if os.path.exists(SPLIT_DIR):
    shutil.rmtree(SPLIT_DIR)
splitDataset()

trainData   = FaceDataset(os.path.join(SPLIT_DIR, "train"), trainTransform)
valData     = FaceDataset(os.path.join(SPLIT_DIR, "val"),   valTransform)
trainLoader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
valLoader   = DataLoader(valData,   batch_size=BATCH_SIZE, shuffle=False)
print(f"Train: {len(trainData)} | Val: {len(valData)}")

# EfficientNet for better accuracy
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 2)
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

bestValAcc = 0

for epoch in range(EPOCHS):
    model.train()
    runningLoss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(trainLoader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    trainAcc = 100 * correct / total
    scheduler.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in valLoader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    valAcc = 100 * correct / total
    print(f"Epoch {epoch+1}: Loss={runningLoss/len(trainLoader):.4f} | Train={trainAcc:.1f}% | Val={valAcc:.1f}%")

    if valAcc > bestValAcc:
        bestValAcc = valAcc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  ✅ Best model saved! Val={valAcc:.1f}%")

print(f"\nTraining complete. Best Val Accuracy: {bestValAcc:.1f}%")