import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

TEST_REAL = os.path.expanduser("~/Desktop/real_vs_fake/real-vs-fake/test/real")
TEST_FAKE = os.path.expanduser("~/Desktop/real_vs_fake/real-vs-fake/test/fake")
MODEL_PATH = "deepfake_detector.pth"
IMG_SIZE   = 224
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
SAMPLE     = 500  # test on 500 images each
print(f"Using device: {DEVICE}")

model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def evaluateFolder(folderPath, trueLabel, limit=SAMPLE):
    images = [f for f in os.listdir(folderPath) if f.endswith((".jpg", ".png"))][:limit]
    correct = 0
    for imgName in tqdm(images, desc=f"Testing {trueLabel}"):
        imgPath = os.path.join(folderPath, imgName)
        try:
            img = Image.open(imgPath).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1)
                _, predicted = torch.max(probs, 1)
            predLabel = "real" if predicted.item() == 0 else "fake"
            if predLabel == trueLabel:
                correct += 1
        except:
            continue
    return correct, len(images)

print("\n📊 EVALUATING ON UNSEEN TEST DATA...")
print(f"Testing {SAMPLE} real + {SAMPLE} fake images\n")

realCorrect, realTotal = evaluateFolder(TEST_REAL, "real")
fakeCorrect, fakeTotal = evaluateFolder(TEST_FAKE, "fake")

totalCorrect = realCorrect + fakeCorrect
totalImages  = realTotal + fakeTotal
accuracy     = 100 * totalCorrect / totalImages

print(f"\n{'='*50}")
print(f"  Real accuracy : {realCorrect}/{realTotal} ({100*realCorrect/realTotal:.1f}%)")
print(f"  Fake accuracy : {fakeCorrect}/{fakeTotal} ({100*fakeCorrect/fakeTotal:.1f}%)")
print(f"  TOTAL ACCURACY: {totalCorrect}/{totalImages} ({accuracy:.1f}%)")
print(f"{'='*50}")
