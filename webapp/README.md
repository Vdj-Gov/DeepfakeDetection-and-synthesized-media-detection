# Demo web UI

Run a small Flask UI to upload a video and get a processed output with face bounding boxes and fake/real labels.

Setup

1. Create a Python virtualenv and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. Run the app:

```bash
python app.py
```

3. Open http://localhost:5000 and upload a video. The app uses OpenCV Haar cascade to detect faces and a deterministic placeholder label (replace `process_video.deterministic_label` with your model call).

Integration notes

- Replace the deterministic labeling in `process_video.py` with a call to your trained detector/classifier and set label/color/score accordingly.
- If you have a GPU-based classifier, call it inside `process_video.process_video` for each face crop or on full frames.
