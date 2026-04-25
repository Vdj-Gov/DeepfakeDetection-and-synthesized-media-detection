import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

TEST_REAL  = os.path.expanduser("~/Desktop/deepFake/real_vs_fake/real-vs-fake/test/real")
TEST_FAKE  = os.path.expanduser("~/Desktop/deepFake/real_vs_fake/real-vs-fake/test/fake")
MODEL_PATH = "deepfake_detector_guided.pth"
IMG_SIZE   = 224
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
SAMPLE     = 500

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
        self.regionWeight = nn.Parameter(
            torch.ones(channels) * (1.0 + 0.1 * numRegions)
        )
    def forward(self, x):
        attn = self.attention(x).unsqueeze(-1).unsqueeze(-1)
        return x * attn * self.regionWeight.unsqueeze(-1).unsqueeze(-1)

class KnowledgeGuidedEfficientNet(nn.Module):
    def __init__(self, numRegions):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(base.classifier[1].in_features, 2)
        )
        self.features   = base.features
        self.avgpool    = base.avgpool
        self.attention  = KnowledgeGuidedAttention(1280, numRegions)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280, 2)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = KnowledgeGuidedEfficientNet(numRegions=7).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def evaluateFolder(folderPath, trueLabel, limit=SAMPLE):
    images = [f for f in os.listdir(folderPath) if f.endswith((".jpg",".png"))][:limit]
    correct = 0
    for imgName in tqdm(images, desc=f"Testing {trueLabel}"):
        imgPath = os.path.join(folderPath, imgName)
        try:
            img = Image.open(imgPath).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(tensor)
                probs  = torch.softmax(output, dim=1)
                _, predicted = torch.max(probs, 1)
            pred = "real" if predicted.item() == 0 else "fake"
            if pred == trueLabel:
                correct += 1
        except:
            continue
    return correct, len(images)

print(f"\n📊 EVALUATING KNOWLEDGE-GUIDED MODEL ON UNSEEN DATA...")
realCorrect, realTotal = evaluateFolder(TEST_REAL, "real")
fakeCorrect, fakeTotal = evaluateFolder(TEST_FAKE, "fake")

total    = realCorrect + fakeCorrect
images   = realTotal + fakeTotal
accuracy = 100 * total / images

print(f"\n{'='*55}")
print(f"  Real accuracy : {realCorrect}/{realTotal} ({100*realCorrect/realTotal:.1f}%)")
print(f"  Fake accuracy : {fakeCorrect}/{fakeTotal} ({100*fakeCorrect/fakeTotal:.1f}%)")
print(f"  TOTAL ACCURACY: {total}/{images} ({accuracy:.1f}%)")
print(f"{'='*55}")
