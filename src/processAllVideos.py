import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
import time


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


# -------- Face Detector --------
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# -------- INPUT & OUTPUT --------
inputFolder = "dataset"
outputFolder = "output_videos_full_2d"

os.makedirs(outputFolder, exist_ok=True)

print("Scanning dataset...")
print("Found structure:", os.listdir(inputFolder))


def format_duration(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def render_bar(progress, width=30):
    progress = max(0.0, min(1.0, progress))
    filled = int(progress * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


videoList = []
for root, _, files in os.walk(inputFolder):
    for videoName in files:
        if videoName.endswith(".mp4"):
            videoList.append((root, videoName))

totalVideos = len(videoList)
if totalVideos == 0:
    print("No videos found.")
    raise SystemExit(0)

print(f"Total videos found: {totalVideos}")
globalStart = time.time()


# -------- Traverse ALL subfolders --------
for idx, (root, videoName) in enumerate(videoList, start=1):
    videoPath = os.path.join(root, videoName)
    print(f"\nProcessing ({idx}/{totalVideos}): {videoPath}")

    cap = cv2.VideoCapture(videoPath)

    if not cap.isOpened():
        print("FAILED TO OPEN:", videoPath)
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # -------- Preserve folder structure --------
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

    frameCount = 0
    videoStart = time.time()
    updateEvery = 30

    # -------- Frame Loop --------
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frameCount += 1

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

                color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                text = f"{label} ({confidence:.2f})"
                cv2.putText(frame, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            except Exception as e:
                print("Error:", e)
                continue

        out.write(frame)

        if totalFrames > 0 and (frameCount % updateEvery == 0 or frameCount == totalFrames):
            elapsed = time.time() - videoStart
            fps = frameCount / elapsed if elapsed > 0 else 0.0
            progress = frameCount / totalFrames
            eta = (totalFrames - frameCount) / fps if fps > 0 else 0
            bar = render_bar(progress)
            msg = (
                f"\r{bar} {progress*100:6.2f}% | "
                f"{fps:6.2f} frames/s | ETA {format_duration(eta)}"
            )
            print(msg, end="", flush=True)

    if totalFrames > 0:
        print()

    cap.release()
    out.release()

    totalElapsed = time.time() - globalStart
    videosPerMin = idx / (totalElapsed / 60.0) if totalElapsed > 0 else 0.0
    remainingVideos = totalVideos - idx
    etaAll = (remainingVideos / videosPerMin) * 60 if videosPerMin > 0 else 0
    print(
        f"SAVED: {outputPath} | Frames: {frameCount} | "
        f"Speed: {videosPerMin:.2f} videos/min | Remaining ETA: {format_duration(etaAll)}"
    )

print("\nAll videos processed successfully!")