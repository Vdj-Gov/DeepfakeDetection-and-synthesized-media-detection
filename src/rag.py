import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import sys
from sentence_transformers import SentenceTransformer, util

def loadKnowledgeBase(path="src/knowledge_base.txt"):
    with open(path, "r") as f:
        content = f.read()
    chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 50]
    return chunks

class RAGRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.embedder.encode(chunks, convert_to_tensor=True)

    def retrieve(self, query, topK=3):
        queryEmbedding = self.embedder.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(queryEmbedding, self.embeddings)[0]
        topIndices = torch.topk(scores, k=topK).indices
        return [self.chunks[i] for i in topIndices]

MODEL_PATH = "deepfake_detector.pth"
IMG_SIZE   = 224
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"

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

def detectImage(imagePath):
    img = Image.open(imagePath).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    label = "REAL" if predicted.item() == 0 else "FAKE"
    return label, confidence.item()

def detectVideo(videoPath):
    cap = cv2.VideoCapture(videoPath)
    fakeCount, realCount, frameCount = 0, 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frameCount % 10 == 0:
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
    confidence = fakeRatio if fakeRatio > 0.5 else (1 - fakeRatio)
    label = "FAKE" if fakeRatio > 0.4 else "REAL"
    return label, confidence, fakeCount, realCount, total

if len(sys.argv) < 2:
    print("Usage: python3 src/rag.py <image_or_video_path>")
    sys.exit(1)

inputPath = sys.argv[1]
ext = os.path.splitext(inputPath)[1].lower()

print("Loading knowledge base...")
chunks = loadKnowledgeBase("src/knowledge_base.txt")
retriever = RAGRetriever(chunks)

if ext in [".mp4", ".avi", ".mov", ".mkv"]:
    label, confidence, fakeCount, realCount, total = detectVideo(inputPath)
    query = f"{label} deepfake video face swap artifacts inconsistencies confidence {confidence*100:.0f}%"
    print(f"\n🎬 Video: {inputPath}")
    print(f"   Frames analyzed : {total}")
    print(f"   Fake frames     : {fakeCount}")
    print(f"   Real frames     : {realCount}")
else:
    label, confidence = detectImage(inputPath)
    query = f"{label} deepfake image artifacts inconsistencies confidence {confidence*100:.0f}%"
    print(f"\n🖼  Image: {inputPath}")

print(f"\n{'='*50}")
print(f"   RESULT     : {label}")
print(f"   CONFIDENCE : {confidence*100:.1f}%")
print(f"{'='*50}")

print("\n📖 ANALYSIS:")
relevantChunks = retriever.retrieve(query, topK=3)
for chunk in relevantChunks:
    print(f"\n{chunk[:500]}")
    print("-" * 40)
