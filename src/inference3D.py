import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import torch
import torchvision.models.video as models
import torchvision.transforms as transforms


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
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# -------- Load Video --------
videoPath = "test.mp4"
cap = cv2.VideoCapture(videoPath)

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()


# -------- Frame Buffer --------
frames = []
clipLength = 16

print("Processing video...")


while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame)

    frames.append(frame)

    # -------- When we have enough frames --------
    if len(frames) == clipLength:

        clip = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(clip)
            prob = torch.softmax(output, dim=1)[0]

        label = "FAKE" if prob[1] > prob[0] else "REAL"

        print(f"Prediction: {label} | Confidence: {prob}")

        frames = []  # reset buffer


cap.release()

print("Done.")