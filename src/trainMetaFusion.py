import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Import all models
from videoDataset import VideoDataset
from models.xceptionModel import createXceptionModel
from models.videoModel import create3DModel
from models.frequencyModel import createFrequencyModel
from models.metaClassifier import createMetaClassifier
from trainFrequency import ExtractFrequencySpectrum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def confidence_features(prob):
    top_vals, _ = torch.topk(prob, k=2, dim=1)
    margin = (top_vals[:, 0] - top_vals[:, 1]).unsqueeze(1)
    max_prob = top_vals[:, 0].unsqueeze(1)
    entropy = -(prob * torch.log(prob.clamp(min=1e-8))).sum(dim=1, keepdim=True)
    return torch.cat([margin, max_prob, entropy], dim=1)

def main():
    print("Loading component models...")
    
    # 1. 2D Model
    model2D = createXceptionModel()
    model2D.load_state_dict(torch.load("models/xception.pth", map_location=device))
    model2D.to(device)
    model2D.eval()

    # 2. 3D Model
    model3D = create3DModel()
    model3D.load_state_dict(torch.load("models/3d_model.pth", map_location=device))
    model3D.to(device)
    model3D.eval()

    # 3. Frequency Model
    modelFreq = createFrequencyModel()
    modelFreq.load_state_dict(torch.load("models/frequency_model.pth", map_location=device))
    modelFreq.to(device)
    modelFreq.eval()

    # Dataloader gives clips of [C, T, H, W]
    dataset = VideoDataset("dataset_subset_8h", clipLength=16, trainMode=False, useRetinaFace=False)
    valSize = max(1, int(0.2 * len(dataset)))
    trainSize = len(dataset) - valSize
    trainSet, valSet = random_split(
        dataset,
        [trainSize, valSize],
        generator=torch.Generator().manual_seed(42),
    )
    trainLoader = DataLoader(trainSet, batch_size=32, shuffle=True)
    valLoader = DataLoader(valSet, batch_size=32, shuffle=False)

    # Initialize Meta Classifier
    # 3 branches x 2 probs + 3 branches x 3 confidence stats = 15 features
    meta_model = createMetaClassifier(input_features=15).to(device)
    optimizer = optim.AdamW(meta_model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()

    freq_extractor = ExtractFrequencySpectrum()

    epochs = 20
    bestValAcc = 0.0
    noImproveCount = 0
    earlyStopPatience = 5
    print("Starting Meta-fusion training...")

    for epoch in range(epochs):
        meta_model.train()
        total_loss = 0
        
        for clips, labels in trainLoader:
            clips = clips.to(device)
            labels = labels.to(device)
            
            # 1. Get 3D network prediction (takes entire clip)
            # clips shape: [Batch, C, T, H, W]
            with torch.no_grad():
                out3D = torch.softmax(model3D(clips), dim=1) # [Batch, 2]

            # 2. Get 2D and Freq network predictions
            # we will average the output over the T frames for 2D models
            B, C, T, H, W = clips.shape
            
            # Reshape to [Batch*T, C, H, W] to run 2D models easily natively
            frames = clips.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            
            with torch.no_grad():
                out2D_frames = torch.softmax(model2D(frames), dim=1) # [Batch*T, 2]
                
                # Freq extraction
                freq_frames = torch.stack([freq_extractor(f.cpu()) for f in frames]).to(device)
                outFreq_frames = torch.softmax(modelFreq(freq_frames), dim=1) # [Batch*T, 2]
                
            # Average over T temporal frames
            out2D = out2D_frames.reshape(B, T, 2).mean(dim=1) # [Batch, 2]
            outFreq = outFreq_frames.reshape(B, T, 2).mean(dim=1) # [Batch, 2]
            
            # Combine calibrated probabilities and confidence summaries
            conf2D = confidence_features(out2D)
            conf3D = confidence_features(out3D)
            confFreq = confidence_features(outFreq)
            meta_features = torch.cat([out2D, out3D, outFreq, conf2D, conf3D, confFreq], dim=1)
            
            # Train Meta Classifier
            optimizer.zero_grad()
            predictions = meta_model(meta_features)
            loss = criterion(predictions, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        meta_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for clips, labels in valLoader:
                clips = clips.to(device)
                labels = labels.to(device)
                out3D = torch.softmax(model3D(clips), dim=1)
                B, C, T, H, W = clips.shape
                frames = clips.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                out2D_frames = torch.softmax(model2D(frames), dim=1)
                freq_frames = torch.stack([freq_extractor(f.cpu()) for f in frames]).to(device)
                outFreq_frames = torch.softmax(modelFreq(freq_frames), dim=1)
                out2D = out2D_frames.reshape(B, T, 2).mean(dim=1)
                outFreq = outFreq_frames.reshape(B, T, 2).mean(dim=1)
                conf2D = confidence_features(out2D)
                conf3D = confidence_features(out3D)
                confFreq = confidence_features(outFreq)
                meta_features = torch.cat([out2D, out3D, outFreq, conf2D, conf3D, confFreq], dim=1)
                predictions = meta_model(meta_features)
                preds = torch.argmax(predictions, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        valAcc = (100.0 * correct / total) if total > 0 else 0.0
        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f} - Val Acc: {valAcc:.2f}%")

        if valAcc > bestValAcc:
            bestValAcc = valAcc
            noImproveCount = 0
            os.makedirs("models", exist_ok=True)
            torch.save(meta_model.state_dict(), "models/meta_classifier.pth")
            print(f"Saved improved meta-classifier (Val Acc: {bestValAcc:.2f}%)")
        else:
            noImproveCount += 1
            if noImproveCount >= earlyStopPatience:
                print("Early stopping triggered for meta-fusion.")
                break

    # Save
    os.makedirs("models", exist_ok=True)
    torch.save(meta_model.state_dict(), "models/meta_classifier_last.pth")
    print(f"Meta Classifier training completed. Best Val Accuracy: {bestValAcc:.2f}%")

if __name__ == "__main__":
    main()
