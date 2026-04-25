import os
import shutil
import random

def create_medium_dataset(source_dir, dest_dir, num_real=500, num_fake=500):
    print("Creating medium dataset...")
    
    os.makedirs(os.path.join(dest_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "fake"), exist_ok=True)

    # Sample REAL
    real_source = os.path.join(source_dir, "original")
    real_videos = [v for v in os.listdir(real_source) if v.endswith(".mp4")]
    sampled_real = random.sample(real_videos, min(num_real, len(real_videos)))
    
    print(f"Sampling {len(sampled_real)} REAL videos...")
    for v in sampled_real:
        shutil.copy(os.path.join(real_source, v), os.path.join(dest_dir, "real", v))

    # Sample FAKE
    fake_dirs = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    all_fake_videos = []
    
    for fd in fake_dirs:
        fd_path = os.path.join(source_dir, fd)
        if os.path.exists(fd_path):
            videos = [os.path.join(fd, v) for v in os.listdir(fd_path) if v.endswith(".mp4")]
            all_fake_videos.extend(videos)
            
    sampled_fake = random.sample(all_fake_videos, min(num_fake, len(all_fake_videos)))
    
    print(f"Sampling {len(sampled_fake)} FAKE videos...")
    for v in sampled_fake:
        # Avoid naming collisions by including parent directory name in filename
        # e.g Deepfakes_001_002.mp4
        safe_name = v.replace(os.sep, "_").replace("/", "_")
        shutil.copy(os.path.join(source_dir, v), os.path.join(dest_dir, "fake", safe_name))

    print(f"Dataset successfully created at {dest_dir}!")

if __name__ == "__main__":
    create_medium_dataset("dataset", "dataset_medium", 100, 100)
