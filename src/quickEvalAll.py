import os
import cv2
import torch
import timm
import torchvision.models.video as video_models
import torchvision.transforms as transforms
from PIL import Image


def center_crop_square(frame):
    h, w = frame.shape[:2]
    size = min(h, w)
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    return frame[y1:y1 + size, x1:x1 + size]


def load_video_frames(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = center_crop_square(frame)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        frames.append((255 * torch.zeros(224, 224, 3)).byte().numpy())
    while len(frames) < max_frames:
        frames.append(frames[-1].copy())
    return frames


def get_video_list(root):
    items = []
    for label_name in ["real", "fake"]:
        folder = os.path.join(root, label_name)
        label = 0 if label_name == "real" else 1
        for name in os.listdir(folder):
            if name.endswith(".mp4"):
                items.append((os.path.join(folder, name), label, name))
    return items


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = "dataset_subset"
    videos = get_video_list(root)
    print(f"Total videos: {len(videos)}")

    tf_2d = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    model2d = timm.create_model("xception", pretrained=False, num_classes=2)
    model2d.load_state_dict(torch.load("models/xception.pth", map_location=device))
    model2d.to(device).eval()

    model3d = video_models.r3d_18(pretrained=False)
    model3d.fc = torch.nn.Linear(model3d.fc.in_features, 2)
    model3d.load_state_dict(torch.load("models/3d_model.pth", map_location=device))
    model3d.to(device).eval()

    correct_2d = 0
    correct_3d = 0
    correct_fusion = 0

    for idx, (path, label, name) in enumerate(videos, start=1):
        frames = load_video_frames(path, max_frames=16)

        tensors_2d = []
        for frame in frames:
            pil = Image.fromarray(frame)
            tensors_2d.append(tf_2d(pil))

        batch_2d = torch.stack(tensors_2d).to(device)
        clip_3d = batch_2d.permute(1, 0, 2, 3).unsqueeze(0)

        with torch.no_grad():
            out2d = torch.softmax(model2d(batch_2d), dim=1).mean(dim=0)
            out3d = torch.softmax(model3d(clip_3d), dim=1).squeeze(0)
            fusion = 0.7 * out2d + 0.3 * out3d

        pred2d = torch.argmax(out2d).item()
        pred3d = torch.argmax(out3d).item()
        predf = torch.argmax(fusion).item()

        correct_2d += int(pred2d == label)
        correct_3d += int(pred3d == label)
        correct_fusion += int(predf == label)

        if idx % 10 == 0 or idx == len(videos):
            print(f"Processed {idx}/{len(videos)}: {name}")

    n = len(videos)
    print(f"2D Quick Accuracy: {100.0 * correct_2d / n:.2f}%")
    print(f"3D Quick Accuracy: {100.0 * correct_3d / n:.2f}%")
    print(f"Fusion Quick Accuracy: {100.0 * correct_fusion / n:.2f}%")


if __name__ == "__main__":
    main()
