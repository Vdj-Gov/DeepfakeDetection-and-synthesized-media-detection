import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import sys
import os

# ── Config ──────────────────────────────────────────────
MODEL_PATH = "deepfake_detector.pth"
IMG_SIZE   = 224
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"

# ── Load model ───────────────────────────────────────────
model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ── Transform ────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── Predict single image ─────────────────────────────────
def predictImage(imagePath):
    img = Image.open(imagePath).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    label = "REAL" if predicted.item() == 0 else "FAKE"
    print(f"\n🖼  Image: {imagePath}")
    print(f"   Result     : {label}")
    print(f"   Confidence : {confidence.item()*100:.1f}%")
    return label, confidence.item()

# ── Predict video ────────────────────────────────────────
def predictVideo(videoPath):
    cap = cv2.VideoCapture(videoPath)
    frameCount = 0
    fakeCount  = 0
    realCount  = 0
    interval   = 10  # check every 10 frames

    print(f"\n🎬 Video: {videoPath}")
    print("   Analyzing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frameCount % interval == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1)
                _, predicted = torch.max(probs, 1)
            if predicted.item() == 0:
                realCount += 1
            else:
                fakeCount += 1
        frameCount += 1

    cap.release()
    total = realCount + fakeCount
    fakeRatio = fakeCount / total if total > 0 else 0
    result = "FAKE" if fakeRatio > 0.4 else "REAL"

    print(f"   Frames analyzed : {total}")
    print(f"   Fake frames     : {fakeCount} ({fakeRatio*100:.1f}%)")
    print(f"   Real frames     : {realCount} ({(1-fakeRatio)*100:.1f}%)")
    print(f"   Result          : {result}")
    return result

# ── Main ─────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python3 src/detect.py <image_or_video_path>")
    sys.exit(1)

inputPath = sys.argv[1]
ext = os.path.splitext(inputPath)[1].lower()

if ext in [".mp4", ".avi", ".mov", ".mkv"]:
    predictVideo(inputPath)
else:
    predictImage(inputPath)