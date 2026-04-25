import os
import cv2
import torch
import timm
import torchvision.models.video as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- Load 2D Model --------
model2D = timm.create_model("xception", pretrained=False, num_classes=2)
model2D.load_state_dict(torch.load("models/xception.pth", map_location=device))
model2D.to(device)
model2D.eval()


# -------- Load MViT (Transformer) Model --------
model3D = models.mvit_v1_b(weights=models.MViT_V1_B_Weights.DEFAULT)
model3D.head[1] = torch.nn.Linear(model3D.head[1].in_features, 2)
model3D.load_state_dict(torch.load("models/mvit_model.pth", map_location=device))
model3D.to(device)
model3D.eval()

# -------- Load CLIP Model --------
import open_clip
clipModel, _, clipPreprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
clipModel.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

realContext = [
    "A real, natural human face with authentic skin texture and perfectly matched lighting.",
    "A genuine, unmanipulated photograph of a person's face.",
    "Authentic human eyes with natural reflections.",
    "A face with natural facial boundaries and consistent shadows."
]
fakeContext = [
    "A GAN-generated artificial face with unnatural skin smoothing.",
    "A deepfake face swap with blurry boundaries and lighting inconsistencies.",
    "A digitally manipulated face with missing shadows and strange eye patterns.",
    "A deepfake video frame showing unnatural facial warping and geometric inconsistencies."
]

textTokens = tokenizer(realContext + fakeContext).to(device)
with torch.no_grad():
    textFeatures = clipModel.encode_text(textTokens)
    textFeatures /= textFeatures.norm(dim=-1, keepdim=True)


# -------- Transform --------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------- Face Detector --------
from retinaface import RetinaFace
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# -------- Settings --------
clipLength = 16
root = "dataset_medium"

print("Warming up RetinaFace (first call can be slow)...")
try:
    _ = RetinaFace.detect_faces(np.zeros((224, 224, 3), dtype=np.uint8))
    print("RetinaFace warmup done.")
except Exception as e:
    print(f"RetinaFace warmup warning: {e}")

videoCount = 0
for labelName in ["real", "fake"]:
    folder = os.path.join(root, labelName)
    for vid in os.listdir(folder):
        if vid.endswith(".mp4"):
            videoCount += 1
print(f"Total videos for fusion evaluation: {videoCount}")


correct = 0
total = 0


# -------- Loop dataset --------
for labelName in ["real", "fake"]:

    folder = os.path.join(root, labelName)
    label = 0 if labelName == "real" else 1

    for vid in os.listdir(folder):

        if not vid.endswith(".mp4"):
            continue

        path = os.path.join(folder, vid)
        print(f"Processing: {vid}")

        cap = cv2.VideoCapture(path)

        faceBuffer = []
        predictions = []

        last3D = torch.tensor([0.5, 0.5]).to(device)

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                continue

            (x, y, w, h) = faces[0]
            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            # -------- 2D --------
            image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            imageTensor = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                out2D = torch.softmax(model2D(imageTensor), dim=1)[0]

            # -------- CLIP Zero-Shot --------
            with torch.no_grad():
                clipImg = clipPreprocess(image).unsqueeze(0).to(device)
                imgFeatures = clipModel.encode_image(clipImg)
                imgFeatures /= imgFeatures.norm(dim=-1, keepdim=True)

                # Similarity scores (scaled by 100 as per CLIP standard)
                textProbs = (100.0 * imgFeatures @ textFeatures.T).softmax(dim=-1)[0]
                
                # First 4 are Real, Last 4 are Fake
                clipReal = textProbs[:len(realContext)].sum()
                clipFake = textProbs[len(realContext):].sum()
                outCLIP = torch.tensor([clipReal, clipFake]).to(device)

            # -------- 3D buffer --------
            faceFrame = transform(face)
            faceBuffer.append(faceFrame)

            if len(faceBuffer) == clipLength:

                clip = torch.stack(faceBuffer).permute(1, 0, 2, 3).unsqueeze(0).to(device)

                with torch.no_grad():
                    last3D = torch.softmax(model3D(clip), dim=1)[0]

                faceBuffer = []

            # -------- 3-Way Tri-Fusion --------
            # 0.45 2D spatial + 0.35 3D temporal + 0.20 CLIP semantic context
            final = 0.45 * out2D + 0.35 * last3D + 0.20 * outCLIP

            pred = torch.argmax(final).item()
            predictions.append(pred)

        cap.release()

        if len(predictions) == 0:
            continue

        # -------- Video-level decision --------
        finalPred = round(sum(predictions) / len(predictions))

        if finalPred == label:
            correct += 1

        total += 1
        if total % 10 == 0:
            runningAcc = (correct / total) * 100
            print(f"Processed {total}/{videoCount} videos - Running Accuracy: {runningAcc:.2f}%")


accuracy = correct / total * 100
print(f"\nFINAL FUSION Accuracy: {accuracy:.2f}%")