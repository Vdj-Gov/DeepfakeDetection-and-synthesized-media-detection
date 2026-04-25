import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DeepfakeDataset(Dataset):

    def __init__(self, rootDir):

        self.imagePaths = []
        self.labels = []

        # Loop through classes: real and fake
        for labelName in ["real", "fake"]:

            classPath = os.path.join(rootDir, labelName)

            # Skip if folder doesn't exist
            if not os.path.exists(classPath):
                continue

            for imgName in os.listdir(classPath):

                # Only load image files
                if imgName.endswith(".jpg") or imgName.endswith(".png"):
                    
                    fullPath = os.path.join(classPath, imgName)
                    self.imagePaths.append(fullPath)

                    # Assign labels
                    if labelName == "real":
                        self.labels.append(0)
                    else:
                        self.labels.append(1)

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ensure correct size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index):

        imagePath = self.imagePaths[index]

        # Load image
        image = Image.open(imagePath).convert("RGB")

        # Apply transforms
        image = self.transform(image)

        label = self.labels[index]

        return image, label