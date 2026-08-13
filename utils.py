import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import time
from collections import deque

# ======================================================
# Configuration
# ======================================================

YOLO_MODEL_PATH = "best.pt"
EMOTION_MODEL_PATH = "best_emotion_model1.keras"

IMAGE_SIZE = 224
FPS_SMOOTHING = 0.90

# ======================================================
# Emotion Labels
# ======================================================

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# ======================================================
# Colors (BGR)
# ======================================================

COLORS = {

    "Angry": (0, 0, 255),

    "Disgust": (0, 140, 255),

    "Fear": (128, 0, 255),

    "Happy": (0, 255, 0),

    "Neutral": (255, 255, 0),

    "Sad": (255, 0, 0),

    "Surprise": (255, 0, 255),

    "Unknown": (180, 180, 180)

}

# ======================================================
# Global Models
# ======================================================

YOLO_MODEL = None
EMOTION_MODEL = None

# ======================================================
# Model Loader
# ======================================================

def load_models():

    global YOLO_MODEL
    global EMOTION_MODEL

    if YOLO_MODEL is None:
        print("Loading YOLO...")
        YOLO_MODEL = YOLO(YOLO_MODEL_PATH)

    if EMOTION_MODEL is None:
        print("Loading Emotion Model...")
        EMOTION_MODEL = tf.keras.models.load_model(EMOTION_MODEL_PATH)

    return YOLO_MODEL, EMOTION_MODEL

# ======================================================
# Face Preprocessing
# ======================================================

def preprocess_face(face):

    face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face = face.astype(np.float32)

    face = face / 255.0

    face = np.expand_dims(face, axis=0)

    return face

# ======================================================
# Emotion Prediction
# ======================================================
# NOTE: emotion_model is grabbed once by realtime.py via load_models()
# at startup already, so calling load_models() here is cheap (it just
# returns the cached globals instead of reloading anything). This keeps
# the function usable standalone without changing realtime.py's calling
# convention.

def predict_emotion(face):

    _, emotion_model = load_models()

    face = preprocess_face(face)

    prediction = emotion_model.predict(face, verbose=0)[0]

    index = np.argmax(prediction)

    emotion = EMOTIONS[index]

    confidence = float(prediction[index])

    return emotion, confidence, prediction

# ======================================================
# Top 3 Prediction
# ======================================================

def top3(prediction):

    indexes = np.argsort(prediction)[::-1][:3]

    result = []

    for idx in indexes:
        result.append((EMOTIONS[idx], float(prediction[idx])))

    return result

# ======================================================
# Draw Box
# ======================================================

def draw_box(

        frame,

        box,

        emotion,

        emotion_conf,

        detect_conf

):

    x1, y1, x2, y2 = box

    color = COLORS.get(emotion, (255, 255, 255))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{emotion} {emotion_conf*100:.1f}%"

    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    detect = f"Face {detect_conf*100:.1f}%"

    cv2.putText(
        frame,
        detect,
        (x1, y2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1
    )

# ======================================================
# FPS Calculator
# ======================================================

class FPS:

    def __init__(self):

        self.prev = time.time()

        self.fps = 0

    def update(self):

        current = time.time()

        delta = current - self.prev

        self.prev = current

        if delta <= 0:
            return int(self.fps)

        new = 1 / delta

        self.fps = (
            FPS_SMOOTHING * self.fps +
            (1 - FPS_SMOOTHING) * new
        )

        return int(self.fps)

# ======================================================
# Emotion Smoothing
# ======================================================

class EmotionSmoother:

    def __init__(self, size=5):

        self.history = deque(maxlen=size)

    def update(self, emotion):

        self.history.append(emotion)

        return max(self.history, key=self.history.count)

# ======================================================
# Draw FPS
# ======================================================

def draw_fps(frame, fps):

    cv2.putText(
        frame,
        f"FPS : {fps}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

# ======================================================
# Draw Title
# ======================================================

def draw_title(frame):

    cv2.putText(
        frame,
        "DeepFER | YOLOv8 + MobileNetV2",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )