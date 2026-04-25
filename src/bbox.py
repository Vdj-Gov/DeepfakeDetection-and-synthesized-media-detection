import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms


# -------- Load Model --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model("xception", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("models/xception.pth", map_location=device))
model.to(device)
model.eval()


# -------- Image Transform --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------- Load Face Detector --------
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# -------- Video Input --------
videoPath = r"C:\Users\Varun\Documents\MinProj\dataset_subset\fake\000_003.mp4"
cap = cv2.VideoCapture(videoPath)

# Check if video opened
if not cap.isOpened():
    print("Error: Cannot open video")
    exit()


# -------- FIXED VideoWriter --------
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "output.avi",
    cv2.VideoWriter_fourcc(*'XVID'),
    20.0,
    (width, height)
)


# -------- Process Frames --------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        try:
            image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                prob = torch.softmax(output, dim=1)[0]

            label = "FAKE" if prob[1] > prob[0] else "REAL"
            confidence = max(prob).item()

            # Draw box
            color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            text = f"{label} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception as e:
            print("Error:", e)
            continue

    # Save frame
    out.write(frame)


# -------- Release --------
cap.release()
out.release()

print("Processing complete. Check output.avi")