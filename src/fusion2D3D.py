import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import torch
import timm
import torchvision.models.video as models
import torchvision.transforms as transforms
from PIL import Image


# -------- Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def confidence_features(prob):
    top_vals, _ = torch.topk(prob, k=2, dim=0)
    margin = (top_vals[0] - top_vals[1]).unsqueeze(0)
    max_prob = top_vals[0].unsqueeze(0)
    entropy = -(prob * torch.log(prob.clamp(min=1e-8))).sum().unsqueeze(0)
    return torch.cat([margin, max_prob, entropy], dim=0)


# -------- Load 2D Model --------
model2D = timm.create_model("xception", pretrained=False, num_classes=2)
model2D.load_state_dict(torch.load("models/xception.pth", map_location=device))
model2D.to(device)
model2D.eval()


# -------- Load 3D Model --------
model3D = models.r3d_18(pretrained=False)
model3D.fc = torch.nn.Linear(model3D.fc.in_features, 2)
model3D.load_state_dict(torch.load("models/3d_model.pth", map_location=device))
model3D.to(device)
model3D.eval()


# -------- Load Frequency Model --------
try:
    from models.frequencyModel import createFrequencyModel
    from trainFrequency import ExtractFrequencySpectrum
    modelFreq = createFrequencyModel()
    if os.path.exists("models/frequency_model.pth"):
        modelFreq.load_state_dict(torch.load("models/frequency_model.pth", map_location=device))
    modelFreq.to(device)
    modelFreq.eval()
    freq_extractor = ExtractFrequencySpectrum()
except Exception as e:
    print(f"Warning: Frequency model load failed: {e}")


# -------- Load Meta Classifier --------
try:
    from models.metaClassifier import createMetaClassifier
    metaClassifier = createMetaClassifier(input_features=15)
    if os.path.exists("models/meta_classifier.pth"):
        metaClassifier.load_state_dict(torch.load("models/meta_classifier.pth", map_location=device))
    metaClassifier.to(device)
    metaClassifier.eval()
except Exception as e:
    print(f"Warning: Meta classifier load failed: {e}")


# -------- Transform --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------- Face Detector --------
from retinaface import RetinaFace


# -------- Paths --------
inputFolder = "dataset"
outputFolder = "output_videos_final_full"

os.makedirs(outputFolder, exist_ok=True)


clipLength = 16


# -------- Traverse dataset --------
for root, dirs, files in os.walk(inputFolder):

    for videoName in files:

        if not videoName.endswith(".mp4"):
            continue

        videoPath = os.path.join(root, videoName)
        print(f"\nProcessing: {videoPath}")

        cap = cv2.VideoCapture(videoPath)

        if not cap.isOpened():
            print("❌ Failed:", videoName)
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Preserve folder structure
        relativePath = os.path.relpath(root, inputFolder)
        saveDir = os.path.join(outputFolder, relativePath)
        os.makedirs(saveDir, exist_ok=True)

        outputPath = os.path.join(saveDir, f"final_{videoName}")

        out = cv2.VideoWriter(
            outputPath,
            cv2.VideoWriter_fourcc(*'XVID'),
            20.0,
            (width, height)
        )

        faceBuffer = []
        last3D = torch.tensor([0.5, 0.5]).to(device)

        lastPrediction = "Processing..."
        lastConfidence = 0.0

        # -------- Frame loop --------
        while True:

            ret, frame = cap.read()
            if not ret:
                break

            faces = RetinaFace.detect_faces(frame)
            
            if not isinstance(faces, dict):
                continue
                
            for key in faces:
                x, y, x2, y2 = faces[key]["facial_area"]
                w, h = x2 - x, y2 - y


                face = frame[y:y+h, x:x+w]

                if face.size == 0:
                    continue

                # -------- 2D --------
                image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                imageTensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    out2D = torch.softmax(model2D(imageTensor), dim=1)[0]
                    
                # -------- Frequency --------
                with torch.no_grad():
                    try:
                        freqTensor = freq_extractor(imageTensor.cpu().squeeze(0)).unsqueeze(0).to(device)
                        outFreq = torch.softmax(modelFreq(freqTensor), dim=1)[0]
                    except:
                        outFreq = torch.tensor([0.5, 0.5]).to(device) # Fallback

                # -------- Add to 3D buffer --------
                faceFrame = transform(image)
                faceBuffer.append(faceFrame)

                # -------- 3D --------
                if len(faceBuffer) == clipLength:

                    clip = torch.stack(faceBuffer).permute(1, 0, 2, 3).unsqueeze(0).to(device)

                    with torch.no_grad():
                        last3D = torch.softmax(model3D(clip), dim=1)[0]

                    faceBuffer = []

                # -------- Fusion --------
                conf2D = confidence_features(out2D)
                conf3D = confidence_features(last3D)
                confFreq = confidence_features(outFreq)
                meta_features = torch.cat([out2D, last3D, outFreq, conf2D, conf3D, confFreq]).unsqueeze(0)
                
                with torch.no_grad():
                    try:
                        final = torch.softmax(metaClassifier(meta_features), dim=1)[0]
                    except:
                        # Fallback if metaClassifier is not loaded/trained yet
                        final = 0.5 * out2D + 0.3 * last3D + 0.2 * outFreq
                        

                lastPrediction = "FAKE" if final[1] > final[0] else "REAL"
                lastConfidence = max(final).item()

                # -------- Draw --------
                color = (0, 0, 255) if lastPrediction == "FAKE" else (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                text = f"{lastPrediction} ({lastConfidence:.2f})"
                cv2.putText(frame, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            out.write(frame)

        cap.release()
        out.release()

        print(f"✅ Saved: {outputPath}")


print("\n🎉 ALL videos processed with FINAL FUSION SYSTEM!")