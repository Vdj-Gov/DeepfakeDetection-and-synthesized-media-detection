import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import shutil
import random

# ── Config ──────────────────────────────────────────────
SPLIT_DIR  = "processed/splits"
MODEL_PATH = "deepfake_detector.pth"
GUIDED_MODEL_PATH = "deepfake_detector_guided.pth"
KB_PATH    = "src/knowledge_base.txt"
EPOCHS     = 10
BATCH_SIZE = 32
LR         = 0.00003
IMG_SIZE   = 224
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── Parse knowledge base for key regions ────────────────
def extractKeyRegions(kbPath):
    with open(kbPath, "r") as f:
        content = f.read().lower()
    regions = {
        "eyes":     "eyes" in content or "blinking" in content,
        "skin":     "skin" in content or "texture" in content,
        "jawline":  "jawline" in content or "jaw" in content,
        "hairline": "hairline" in content or "hair" in content,
        "teeth":    "teeth" in content,
        "lighting": "lighting" in content or "shadows" in content,
        "boundaries": "boundaries" in content or "seam" in content,
    }
    detected = [r for r, found in regions.items() if found]
    print(f"\n📖 Knowledge base detected these key regions to focus on:")
    for r in detected:
        print(f"   - {r}")
    return detected

# ── Attention module guided by knowledge base ────────────
class KnowledgeGuidedAttention(nn.Module):
    def __init__(self, channels, numRegions):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        # Region weights based on knowledge base
        self.regionWeight = nn.Parameter(
            torch.ones(channels) * (1.0 + 0.1 * numRegions)
        )

    def forward(self, x):
        attn = self.attention(x).unsqueeze(-1).unsqueeze(-1)
        return x * attn * self.regionWeight.unsqueeze(-1).unsqueeze(-1)

# ── Knowledge guided model ───────────────────────────────
class KnowledgeGuidedEfficientNet(nn.Module):
    def __init__(self, numRegions):
        super().__init__()
        base = models.efficientnet_b0(weights="IMAGENET1K_V1")

        # Load pretrained deepfake weights
        pretrained = models.efficientnet_b0(weights=None)
        pretrained.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(pretrained.classifier[1].in_features, 2)
        )
        pretrained.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

        self.features  = pretrained.features
        self.avgpool   = pretrained.avgpool
        self.attention = KnowledgeGuidedAttention(1280, numRegions)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)   # Knowledge-guided attention applied here
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ── Dataset ──────────────────────────────────────────────
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

# ── Transforms ───────────────────────────────────────────
trainTransform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
valTransform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── Main ─────────────────────────────────────────────────
print("\n📖 Reading knowledge base...")
keyRegions = extractKeyRegions(KB_PATH)
numRegions = len(keyRegions)

trainData   = FaceDataset(os.path.join(SPLIT_DIR, "train"), trainTransform)
valData     = FaceDataset(os.path.join(SPLIT_DIR, "val"),   valTransform)
trainLoader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
valLoader   = DataLoader(valData,   batch_size=BATCH_SIZE, shuffle=False)
print(f"Train: {len(trainData)} | Val: {len(valData)}")

print(f"\n🧠 Building knowledge-guided model with {numRegions} region attention...")
model = KnowledgeGuidedEfficientNet(numRegions).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

bestValAcc = 0

print("\n🚀 Training with knowledge-guided attention...\n")
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
        torch.save(model.state_dict(), GUIDED_MODEL_PATH)
        print(f"  ✅ Best model saved! Val={valAcc:.1f}%")

print(f"\n✅ Training complete. Best Val Accuracy: {bestValAcc:.1f}%")
print(f"✅ Model saved to {GUIDED_MODEL_PATH}")