import os
import shutil
import random

realPath = "processed_medium/faces/real"
fakePath = "processed_medium/faces/fake"

outputRoot = "processed_medium/splits"

trainRatio = 0.8


def splitClass(inputPath, className):
    allImages = []

    # Collect all images from all video folders
    for folder in os.listdir(inputPath):
        folderPath = os.path.join(inputPath, folder)
        for img in os.listdir(folderPath):
            allImages.append(os.path.join(folderPath, img))

    random.shuffle(allImages)

    splitIndex = int(len(allImages) * trainRatio)

    trainImages = allImages[:splitIndex]
    valImages = allImages[splitIndex:]

    for imgPath in trainImages:
        destFolder = os.path.join(outputRoot, "train", className)
        os.makedirs(destFolder, exist_ok=True)
        shutil.copy(imgPath, destFolder)

    for imgPath in valImages:
        destFolder = os.path.join(outputRoot, "val", className)
        os.makedirs(destFolder, exist_ok=True)
        shutil.copy(imgPath, destFolder)


print("Splitting REAL...")
splitClass(realPath, "real")

print("Splitting FAKE...")
splitClass(fakePath, "fake")

print("Dataset split complete.")