import os
from tqdm import tqdm
from frameExtractor import extractFrames

realPath = "dataset_subset/real"
fakePath = "dataset_subset/fake"

outputReal = "processed/frames/real"
outputFake = "processed/frames/fake"

def processFolder(inputPath, outputPath):
    os.makedirs(outputPath, exist_ok=True)

    videos = [v for v in os.listdir(inputPath) if v.endswith(".mp4")]

    for video in tqdm(videos):
        videoPath = os.path.join(inputPath, video)
        videoName = video[:-4]

        outputFolder = os.path.join(outputPath, videoName)

        extractFrames(videoPath, outputFolder, fps=2)

print("Processing REAL videos...")
processFolder(realPath, outputReal)

print("Processing FAKE videos...")
processFolder(fakePath, outputFake)

print("Frame extraction complete.")