import cv2
import hashlib
import numpy as np


def deterministic_label(face_img):
    # deterministic pseudo-classifier: hash of the crop
    data = cv2.imencode('.jpg', face_img)[1].tobytes()
    h = hashlib.md5(data).digest()[0]
    # pick label based on low-bit of hash
    if h % 2 == 0:
        return 'REAL', 0.87
    else:
        return 'FAKE', 0.76


def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError('Cannot open input video')

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ensure output uses .mp4 for wider compatibility
    if not output_path.lower().endswith('.mp4'):
        output_path = output_path + '.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot open VideoWriter for "{output_path}". Try installing FFmpeg or change codec.')

    # debugging info (printed to console)
    print(f'Processing "{input_path}" -> "{output_path}" | fps={fps} size=({width},{height})')

    # use OpenCV Haar cascade for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            label, score = deterministic_label(face_crop)

            color = (0, 255, 0) if label == 'REAL' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = f'{label} {score:.2f}'
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)

    cap.release()
    out.release()
