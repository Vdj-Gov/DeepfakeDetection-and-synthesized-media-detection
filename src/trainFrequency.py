print("TRAIN FREQUENCY SCRIPT STARTED")
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from datasetLoader import DeepfakeDataset
from models.frequencyModel import createFrequencyModel

# -------- Custom Transform for Frequency Domain (DFT) --------
class ExtractFrequencySpectrum:
    def __call__(self, img_tensor):
        # img_tensor is [3, H, W] from the DeepfakeDataset standard transform
        # We'll convert it to grayscale
        gray = img_tensor.mean(dim=0).numpy()
        
        # Apply 2D Discrete Fourier Transform
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Get magnitude spectrum
        magnitude = 20 * np.log(np.abs(f_shift) + 1)
        
        # Normalize between 0 and 1
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        # Return as [1, H, W]
        freq_tensor = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)
        return freq_tensor

# Overwrite the dataset transform temporarily, or create a custom one inside the loop
# We will intercept the dataloader outputs and apply the frequency transform dynamically
def apply_frequency_transform(images):
    freq_transformer = ExtractFrequencySpectrum()
    freq_images = torch.stack([freq_transformer(img) for img in images])
    return freq_images


# -------- Setup --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

trainDataset = DeepfakeDataset("processed/splits/train")
valDataset = DeepfakeDataset("processed/splits/val")

print("Train size:", len(trainDataset))
print("Val size:", len(valDataset))

trainLoader = DataLoader(trainDataset, batch_size=8, shuffle=True)
valLoader = DataLoader(valDataset, batch_size=8)

model = createFrequencyModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 5

# -------- Training Loop --------
for epoch in range(epochs):
    model.train()
    totalLoss = 0

    for images, labels in trainLoader:
        # Transform the standard RGB images into Frequency Spectrums
        freq_images = apply_frequency_transform(images).to(device)
        labels = labels.to(device)

        outputs = model(freq_images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()

    print(f"Epoch {epoch+1} Loss: {totalLoss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valLoader:
            freq_images = apply_frequency_transform(images).to(device)
            labels = labels.to(device)

            outputs = model(freq_images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/frequency_model.pth")
print("Frequency Model saved successfully")
