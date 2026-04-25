import cv2
import os

def extractFrames(videoPath, outputFolder, fps=2):
    os.makedirs(outputFolder, exist_ok=True)

    cap = cv2.VideoCapture(videoPath)
    videoFps = cap.get(cv2.CAP_PROP_FPS)

    if videoFps == 0:
        print(f"Skipping {videoPath} (Invalid FPS)")
        return

    frameInterval = int(videoFps / fps)

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frameInterval == 0:
            frameName = f"frame_{saved}.jpg"
            framePath = os.path.join(outputFolder, frameName)
            cv2.imwrite(framePath, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"Extracted {saved} frames from {videoPath}")