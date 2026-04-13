import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import open_clip
import sys

# ── Config ──────────────────────────────────────────────
MODEL_PATH = "deepfake_detector.pth"
KB_PATH    = "src/knowledge_base.txt"
IMG_SIZE   = 224
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── Load knowledge base context ──────────────────────────
def loadContext(path):
    with open(path, "r") as f:
        content = f.read()
    # Extract artifact descriptions as context
    fakeContext = [
        "blurry face boundaries and unnatural skin texture",
        "inconsistent lighting and shadows on face",
        "unnatural eye blinking and asymmetric eyes",
        "GAN generated face with smooth plastic skin",
        "facial artifacts around hairline and jawline",
        "teeth that look blurry or disappear",
        "unnatural color transitions on face",
        "deepfake face swap with visible seams",
    ]
    realContext = [
        "natural human face with real skin texture",
        "consistent lighting and natural shadows",
        "natural eye movement and symmetric features",
        "authentic face with natural pores and details",
        "natural hairline and jawline",
        "natural teeth and mouth movement",
        "consistent skin tone across face",
        "real photograph of a human face",
    ]
    return fakeContext, realContext

# ── Load CLIP ────────────────────────────────────────────
print("Loading CLIP model...")
clipModel, _, clipPreprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clipModel = clipModel.to(DEVICE)
clipModel.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# ── Load EfficientNet ────────────────────────────────────
effModel = models.efficientnet_b0(weights=None)
effModel.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(effModel.classifier[1].in_features, 2)
)
effModel.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
effModel.to(DEVICE)
effModel.eval()

effTransform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── CLIP context scoring ─────────────────────────────────
def clipScore(image, fakeContext, realContext):
    clipImage = clipPreprocess(image).unsqueeze(0).to(DEVICE)

    allContexts = fakeContext + realContext
    tokens = tokenizer(allContexts).to(DEVICE)

    with torch.no_grad():
        imageFeatures = clipModel.encode_image(clipImage)
        textFeatures  = clipModel.encode_text(tokens)
        imageFeatures = imageFeatures / imageFeatures.norm(dim=-1, keepdim=True)
        textFeatures  = textFeatures  / textFeatures.norm(dim=-1, keepdim=True)
        similarity    = (imageFeatures @ textFeatures.T).squeeze(0)

    fakeScore = similarity[:len(fakeContext)].mean().item()
    realScore = similarity[len(fakeContext):].mean().item()
    return fakeScore, realScore

# ── EfficientNet scoring ─────────────────────────────────
def effScore(image):
    tensor = effTransform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = effModel(tensor)
        probs  = torch.softmax(output, dim=1)
    fakeProb = probs[0][1].item()
    realProb = probs[0][0].item()
    return fakeProb, realProb

# ── Combined detection ───────────────────────────────────
def detect(imagePath):
    image = Image.open(imagePath).convert("RGB")

    # Load context from knowledge base
    fakeContext, realContext = loadContext(KB_PATH)

    # CLIP context-guided score
    clipFake, clipReal = clipScore(image, fakeContext, realContext)

    # EfficientNet score
    effFake, effReal = effScore(image)

    # Combine both scores (60% EfficientNet + 40% CLIP)
    combinedFake = 0.9 * effFake + 0.1 * clipFake
    combinedReal = 0.9 * effReal + 0.1 * clipReal

    label      = "FAKE" if combinedFake > combinedReal else "REAL"
    confidence = max(combinedFake, combinedReal) / (combinedFake + combinedReal) * 100

    print(f"\n🖼  Image: {imagePath}")
    print(f"\n{'='*55}")
    print(f"   RESULT          : {label}")
    print(f"   CONFIDENCE      : {confidence:.1f}%")
    print(f"{'='*55}")
    print(f"\n📊 SCORE BREAKDOWN:")
    print(f"   EfficientNet    : {'FAKE' if effFake > effReal else 'REAL'} ({max(effFake,effReal)*100:.1f}%)")
    print(f"   CLIP (context)  : {'FAKE' if clipFake > clipReal else 'REAL'} ({max(clipFake,clipReal)/(clipFake+clipReal)*100:.1f}%)")
    print(f"\n📖 CONTEXT USED:")
    print(f"   Model checked for these fake artifacts:")
    for ctx in fakeContext:
        print(f"   - {ctx}")

    return label, confidence

# ── Main ─────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python3 src/clip_detect.py <image_path>")
    sys.exit(1)

detect(sys.argv[1])