print("TRAIN SCRIPT STARTED")
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasetLoader import DeepfakeDataset
from models.xceptionModel import createXceptionModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


trainDataset = DeepfakeDataset("processed_medium/splits/train")
valDataset = DeepfakeDataset("processed_medium/splits/val")

print("Train size:", len(trainDataset))
print("Val size:", len(valDataset))


trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
valLoader = DataLoader(valDataset, batch_size=32)


model = createXceptionModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 5


for epoch in range(epochs):

    model.train()
    totalLoss = 0

    for images, labels in trainLoader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
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

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")
torch.save(model.state_dict(), "models/xception.pth")
print("Model saved successfully")