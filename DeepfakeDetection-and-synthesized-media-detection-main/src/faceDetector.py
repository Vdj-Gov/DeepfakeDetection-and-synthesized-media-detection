import os
import cv2
from retinaface import RetinaFace
from tqdm import tqdm


def detectAndSaveFaces(inputFolder, outputFolder):
    os.makedirs(outputFolder, exist_ok=True)  # Create output folder if missing

    images = [img for img in os.listdir(inputFolder) if img.endswith(".jpg")]

    for imageName in tqdm(images):
        imagePath = os.path.join(inputFolder, imageName)
        image = cv2.imread(imagePath)

        # Detect faces using RetinaFace
        faces = RetinaFace.detect_faces(image)

        # If no faces detected, skip image
        if not faces:
            continue

        # Iterate over detected faces (usually 1 in FaceForensics)
        for key in faces:
            faceData = faces[key]
            x1, y1, x2, y2 = faceData["facial_area"]

            # Crop face region
            face = image[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # Resize to 224x224 (model input size)
            face = cv2.resize(face, (224, 224))

            savePath = os.path.join(outputFolder, imageName)
            cv2.imwrite(savePath, face)

            break  # Only save the first detected face