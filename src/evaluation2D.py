import os
import cv2
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- Load model --------
model = timm.create_model("xception", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("models/xception.pth", map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------- Dataset --------
root = "dataset_subset_8h"


correct = 0
total = 0
totalVideosPlanned = 0

for labelName in ["real", "fake"]:
    folder = os.path.join(root, labelName)
    if os.path.isdir(folder):
        totalVideosPlanned += len([f for f in os.listdir(folder) if f.endswith(".mp4")])

print(f"Evaluating videos: {totalVideosPlanned}")

for labelName in ["real", "fake"]:

    folder = os.path.join(root, labelName)
    label = 0 if labelName == "real" else 1

    for fileName in os.listdir(folder):
        if not fileName.endswith(".mp4"):
            continue

        path = os.path.join(folder, fileName)
        cap = cv2.VideoCapture(path)
        framePreds = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                pred = torch.argmax(output, dim=1).item()
                framePreds.append(pred)

        cap.release()

        if len(framePreds) == 0:
            continue

        # Video-level majority vote
        videoPred = round(sum(framePreds) / len(framePreds))
        if videoPred == label:
            correct += 1
        total += 1
        if total % 20 == 0 or total == totalVideosPlanned:
            runningAcc = 100.0 * correct / total
            print(f"Processed {total}/{totalVideosPlanned} videos | Running Accuracy: {runningAcc:.2f}%")


accuracy = (correct / total * 100) if total > 0 else 0.0
print(f"2D Model Accuracy: {accuracy:.2f}%")