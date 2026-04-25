import os
from tqdm import tqdm
from faceDetector import detectAndSaveFaces

realFramesPath = "processed_medium/frames/real"
fakeFramesPath = "processed_medium/frames/fake"

outputRealFaces = "processed_medium/faces/real"
outputFakeFaces = "processed_medium/faces/fake"

def processFaceFolder(framesRoot, outputRoot):
    os.makedirs(outputRoot, exist_ok=True)

    videoFolders = os.listdir(framesRoot)

    for folder in tqdm(videoFolders):
        inputFolder = os.path.join(framesRoot, folder)
        outputFolder = os.path.join(outputRoot, folder)

        detectAndSaveFaces(inputFolder, outputFolder)

print("Processing REAL faces...")
processFaceFolder(realFramesPath, outputRealFaces)

print("Processing FAKE faces...")
processFaceFolder(fakeFramesPath, outputFakeFaces)

print("Face extraction complete.")