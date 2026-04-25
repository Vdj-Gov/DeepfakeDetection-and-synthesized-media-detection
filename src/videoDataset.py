import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random


class VideoDataset(Dataset):

    def __init__(
        self,
        rootDir,
        clipLength=16,
        samples=None,
        trainMode=False,
        useRetinaFace=True,
    ):

        self.samples = []
        self.clipLength = clipLength
        self.trainMode = trainMode
        self.useRetinaFace = useRetinaFace

        if samples is not None:
            self.samples = samples
        else:
            realFolder = os.path.join(rootDir, "real")
            fakeFolder = os.path.join(rootDir, "fake")
            if os.path.isdir(realFolder) and os.path.isdir(fakeFolder):
                for labelName in ["real", "fake"]:
                    folder = os.path.join(rootDir, labelName)
                    for vid in os.listdir(folder):
                        if vid.endswith(".mp4"):
                            path = os.path.join(folder, vid)
                            label = 0 if labelName == "real" else 1
                            self.samples.append((path, label))
            else:
                # Full dataset layout: "original" is real, all other video folders are fake.
                for labelName in os.listdir(rootDir):
                    folder = os.path.join(rootDir, labelName)
                    if not os.path.isdir(folder) or labelName.lower() == "csv":
                        continue
                    label = 0 if labelName.lower() == "original" else 1
                    for vid in os.listdir(folder):
                        if vid.endswith(".mp4"):
                            path = os.path.join(folder, vid)
                            self.samples.append((path, label))

        # -------- Transform (IMPORTANT: normalized) --------
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.augTransform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((240, 240)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # -------- Face detector --------
        self.faceCascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _center_crop_face_fallback(frame):
        h, w = frame.shape[:2]
        size = min(h, w)
        x = (w - size) // 2
        y = (h - size) // 2
        return frame[y:y + size, x:x + size]

    def _extract_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = frame[y:y+h, x:x+w]
            if face.size > 0:
                return face

        # If absolutely no face found, drop to fallback so 3D clip length doesn't crash
        return self._center_crop_face_fallback(frame)

    def _sample_frames(self, videoPath, stride=3):
        cap = cv2.VideoCapture(videoPath)
        rawFrames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rawFrames.append(frame)
        cap.release()

        if len(rawFrames) == 0:
            return []

        # Stride sampling: pick every 'stride' frame from a random start point
        required_span = self.clipLength * stride
        
        if self.trainMode and len(rawFrames) > required_span:
            start = random.randint(0, len(rawFrames) - required_span)
            return rawFrames[start : start + required_span : stride]
        elif len(rawFrames) > required_span:
            return rawFrames[0 : required_span : stride]
        
        # Fallback if video is too short for a strided sequence
        if self.trainMode and len(rawFrames) > self.clipLength:
            start = random.randint(0, len(rawFrames) - self.clipLength)
            return rawFrames[start:start + self.clipLength]
            
        return rawFrames[:self.clipLength]

    def __getitem__(self, idx):

        videoPath, label = self.samples[idx]

        faceFrames = []
        rawFrames = self._sample_frames(videoPath)

        for frame in rawFrames:
            face = self._extract_face(frame)
            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            if self.trainMode:
                face = self.augTransform(face)
            else:
                face = self.transform(face)
            faceFrames.append(face)

        # -------- If no faces detected --------
        if len(faceFrames) == 0:
            # fallback: use blank frame
            faceFrames.append(torch.zeros(3, 224, 224))

        # -------- Pad if needed --------
        while len(faceFrames) < self.clipLength:
            faceFrames.append(faceFrames[-1])

        # -------- Convert shape --------
        # (T, C, H, W) → (C, T, H, W)
        faceFrames = torch.stack(faceFrames).permute(1, 0, 2, 3)

        return faceFrames, torch.tensor(label)