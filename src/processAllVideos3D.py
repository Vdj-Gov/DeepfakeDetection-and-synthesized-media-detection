import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import torch
import torchvision.models.video as models
import torchvision.transforms as transforms


# -------- Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- Load Model --------
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


# -------- Input / Output --------
inputFolder = "dataset_subset"
outputFile = "3d_results.txt"

clipLength = 16

results = []


# -------- Traverse Dataset --------
for root, dirs, files in os.walk(inputFolder):

    for videoName in files:

        if not videoName.endswith(".mp4"):
            continue

        videoPath = os.path.join(root, videoName)
        print(f"Processing: {videoPath}")

        cap = cv2.VideoCapture(videoPath)

        if not cap.isOpened():
            print("Failed:", videoPath)
            continue

        frames = []
        predictions = []

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)

            frames.append(frame)

            # -------- Run on clip --------
            if len(frames) == clipLength:

                clip = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(clip)
                    prob = torch.softmax(output, dim=1)[0]

                predictions.append(prob)

                frames = []

        cap.release()

        # -------- Aggregate Predictions --------
        if len(predictions) == 0:
            print("No clips found:", videoName)
            continue

        avgPred = torch.stack(predictions).mean(dim=0)

        label = "FAKE" if avgPred[1] > avgPred[0] else "REAL"

        resultLine = f"{videoName} → {label} | {avgPred.tolist()}"
        print(resultLine)

        results.append(resultLine)


# -------- Save Results --------
with open(outputFile, "w") as f:
    for line in results:
        f.write(line + "\n")


print("\nAll videos processed. Results saved to 3d_results.txt")