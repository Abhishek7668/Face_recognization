import cv2
import os
import time
import numpy as np

from utils import (
    load_models,
    predict_emotion,
    draw_box,
    draw_fps,
    draw_title,
    FPS,
    EmotionSmoother,
    top3
)

# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_ID = 0

YOLO_CONF = 0.45

YOLO_IMAGE_SIZE = 320

FRAME_SKIP = 3

EMOTION_SKIP = 5

FACE_PADDING = 10

MAX_FACES = 3

WINDOW_NAME = "DeepFER"

OUTPUT_FOLDER = "outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# LOAD MODELS
# ============================================================

print("Loading Models...")

yolo_model, emotion_model = load_models()

print("Models Loaded Successfully")

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise Exception("Cannot open webcam")

FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ============================================================
# VARIABLES
# ============================================================

fps_counter = FPS()

frame_number = 0

cached_boxes = []

emotion_cache = {}

face_smoothers = {}

recording = False

video_writer = None

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

print("\n===============================")
print("DeepFER Started")
print("===============================")
print("Q : Quit")
print("S : Screenshot")
print("R : Start / Stop Recording")
print("===============================\n")

# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        success, frame = cap.read()

        if not success:
            print("Failed to grab frame, exiting...")
            break

        frame_number += 1

        # ====================================================
        # YOLO DETECTION (runs only every FRAME_SKIP frames)
        # ====================================================

        if frame_number % FRAME_SKIP == 0:

            results = yolo_model.predict(
                source=frame,
                imgsz=YOLO_IMAGE_SIZE,
                conf=YOLO_CONF,
                verbose=False
            )

            cached_boxes = []

            for result in results:
                for box in result.boxes:
                    cached_boxes.append(box)

        # ====================================================
        # FACE LOOP
        # ====================================================

        face_id = 0

        for box in cached_boxes:

            if face_id >= MAX_FACES:
                break

            face_id += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detect_conf = float(box.conf[0])

            x1 = max(0, x1 - FACE_PADDING)
            y1 = max(0, y1 - FACE_PADDING)

            x2 = min(frame.shape[1], x2 + FACE_PADDING)
            y2 = min(frame.shape[0], y2 + FACE_PADDING)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # ====================================================
            # Create Emotion Smoother for New Face
            # ====================================================

            if face_id not in face_smoothers:
                face_smoothers[face_id] = EmotionSmoother(size=5)

            # ====================================================
            # Emotion Prediction (runs only every EMOTION_SKIP frames
            # or on first sighting of a face id)
            # ====================================================

            if (
                face_id not in emotion_cache
                or frame_number % EMOTION_SKIP == 0
            ):

                start = time.time()

                emotion, emotion_conf, prediction = predict_emotion(face)

                inference_time = (time.time() - start) * 1000

                emotion = face_smoothers[face_id].update(emotion)

                emotion_cache[face_id] = {
                    "emotion": emotion,
                    "confidence": emotion_conf,
                    "prediction": prediction,
                    "time": inference_time
                }

            data = emotion_cache[face_id]

            emotion = data["emotion"]
            emotion_conf = data["confidence"]
            prediction = data["prediction"]
            inference_time = data["time"]

            # ====================================================
            # Draw Face
            # ====================================================

            draw_box(
                frame,
                (x1, y1, x2, y2),
                emotion,
                emotion_conf,
                detect_conf
            )

            # ====================================================
            # Face ID
            # ====================================================

            cv2.putText(
                frame,
                f"Face #{face_id}",
                (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            # ====================================================
            # Inference Time
            # ====================================================

            cv2.putText(
                frame,
                f"{inference_time:.1f} ms",
                (x1, y2 + 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )

            # ====================================================
            # Top-3 Emotion (Console)
            # ====================================================

            if frame_number % 30 == 0:

                print("\n----------------------")
                print(f"Face #{face_id}")

                for emo, score in top3(prediction):
                    print(f"{emo:<10} : {score*100:.2f}%")

                print("----------------------")

        # ====================================================
        # No faces cleanup: drop stale cache entries so old
        # emotions don't linger forever if a face leaves frame
        # ====================================================

        active_ids = set(range(1, min(len(cached_boxes), MAX_FACES) + 1))

        for stale_id in list(emotion_cache.keys()):
            if stale_id not in active_ids:
                emotion_cache.pop(stale_id, None)
                face_smoothers.pop(stale_id, None)

        # ====================================================
        # FPS + TITLE OVERLAY
        # ====================================================

        current_fps = fps_counter.update()

        draw_fps(frame, current_fps)
        draw_title(frame)

        # ====================================================
        # RECORDING (write frame if active)
        # ====================================================

        if recording and video_writer is not None:
            video_writer.write(frame)

            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "REC",
                (frame.shape[1] - 80, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        # ====================================================
        # QUIT
        # ====================================================

        if key == ord('q'):
            print("Quitting...")
            break

        # ====================================================
        # SCREENSHOT
        # ====================================================

        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(OUTPUT_FOLDER, f"screenshot_{timestamp}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

        # ====================================================
        # START / STOP RECORDING
        # ====================================================

        elif key == ord('r'):

            if not recording:

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(OUTPUT_FOLDER, f"recording_{timestamp}.mp4")

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                video_writer = cv2.VideoWriter(
                    filename,
                    fourcc,
                    20.0,
                    (FRAME_W, FRAME_H)
                )

                recording = True

                print(f"Recording started: {filename}")

            else:

                recording = False

                if video_writer is not None:
                    video_writer.release()
                    video_writer = None

                print("Recording stopped")

finally:

    # ====================================================
    # CLEANUP
    # ====================================================

    if video_writer is not None:
        video_writer.release()

    cap.release()
    cv2.destroyAllWindows()

    print("Resources released. Bye!")