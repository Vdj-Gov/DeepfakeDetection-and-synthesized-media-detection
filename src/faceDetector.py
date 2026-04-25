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

        # Detect faces using Haar Cascades
        faceCascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        # If no faces detected, skip image
        if len(faces) == 0:
            continue

        # Iterate over detected faces (usually 1 in FaceForensics)
        for (x, y, w, h) in faces:
            # Crop face region
            face = image[y:y+h, x:x+w]

            if face.size == 0:
                continue

            # Resize to 224x224 (model input size)
            face = cv2.resize(face, (224, 224))

            savePath = os.path.join(outputFolder, imageName)
            cv2.imwrite(savePath, face)

            break  # Only save the first detected face