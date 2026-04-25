import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import torch
import torchvision.models.video as models
import torchvision.transforms as transforms
from PIL import Image


# -------- Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- Load 3D Model --------
model = models.r3d_18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("models/3d_model.pth", map_location=device))
model.to(device)
model.eval()


# -------- Transform --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------- Face Detector --------
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# -------- Paths --------
inputFolder = "dataset_subset"
outputFolder = "output_videos_3d"

os.makedirs(outputFolder, exist_ok=True)


clipLength = 16


# -------- Traverse dataset --------
for root, dirs, files in os.walk(inputFolder):

    for videoName in files:

        if not videoName.endswith(".mp4"):
            continue

        videoPath = os.path.join(root, videoName)
        print(f"\nProcessing: {videoPath}")

        cap = cv2.VideoCapture(videoPath)

        if not cap.isOpened():
            print("❌ Failed to open:", videoName)
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Preserve folder structure
        relativePath = os.path.relpath(root, inputFolder)
        saveDir = os.path.join(outputFolder, relativePath)
        os.makedirs(saveDir, exist_ok=True)

        outputPath = os.path.join(saveDir, f"out_{videoName}")

        out = cv2.VideoWriter(
            outputPath,
            cv2.VideoWriter_fourcc(*'XVID'),
            20.0,
            (width, height)
        )

        faceBuffer = []
        lastPrediction = "Processing..."
        lastConfidence = 0.0

        # -------- Frame loop --------
        while True:

            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)

            # -------- Collect face frames --------
            if len(faces) > 0:

                (x, y, w, h) = faces[0]
                face = frame[y:y+h, x:x+w]

                if face.size != 0:

                    image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    image = transform(image)

                    faceBuffer.append(image)

            # -------- Run 3D CNN --------
            if len(faceBuffer) == clipLength:

                clip = torch.stack(faceBuffer).permute(1, 0, 2, 3).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(clip)
                    prob = torch.softmax(output, dim=1)[0]

                lastPrediction = "FAKE" if prob[1] > prob[0] else "REAL"
                lastConfidence = max(prob).item()

                faceBuffer = []

            # -------- Draw --------
            for (x, y, w, h) in faces:

                color = (0, 0, 255) if lastPrediction == "FAKE" else (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                text = f"{lastPrediction} ({lastConfidence:.2f})"
                cv2.putText(frame, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            out.write(frame)

        cap.release()
        out.release()

        print(f"✅ Saved: {outputPath}")


print("\n🎉 All videos processed with 3D CNN + BBOX!")