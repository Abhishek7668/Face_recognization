import cv2
import time
import numpy as np
import streamlit as st

from utils import (
    load_models,
    predict_emotion,
    draw_box,
    draw_fps,
    draw_title,
    FPS,
    EmotionSmoother,
    top3,
    EMOTIONS
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DeepFER",
    page_icon="😊",
    layout="wide"
)

# ============================================================
# CONFIG
# ============================================================

YOLO_CONF = 0.45
YOLO_IMAGE_SIZE = 320
FACE_PADDING = 10
MAX_FACES = 3

FRAME_SKIP = 3
EMOTION_SKIP = 5

# ============================================================
# LOAD MODELS (cached so it only loads once per session)
# ============================================================

@st.cache_resource
def get_models():
    return load_models()

with st.spinner("Loading models..."):
    yolo_model, emotion_model = get_models()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("DeepFER")
st.sidebar.caption("YOLOv8 + MobileNetV2 Emotion Detection")

mode = st.sidebar.radio(
    "Choose Mode",
    ["📷 Upload Photo", "🎥 Live Webcam"]
)

yolo_conf = st.sidebar.slider("Detection Confidence", 0.1, 0.9, YOLO_CONF, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**Legend**")
for emo in EMOTIONS:
    st.sidebar.markdown(f"- {emo}")

# ============================================================
# SHARED: RUN DETECTION + EMOTION ON A SINGLE FRAME
# ============================================================

def process_frame(frame, conf, smoothers=None, smoother_key_prefix=""):
    """
    Runs YOLO face detection + emotion prediction on a BGR frame.
    Returns annotated frame and a list of per-face result dicts.
    """

    results = yolo_model.predict(
        source=frame,
        imgsz=YOLO_IMAGE_SIZE,
        conf=conf,
        verbose=False
    )

    face_results = []
    face_id = 0

    for result in results:
        for box in result.boxes:

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

            emotion, emotion_conf, prediction = predict_emotion(face)

            if smoothers is not None:
                key = f"{smoother_key_prefix}{face_id}"
                if key not in smoothers:
                    smoothers[key] = EmotionSmoother(size=5)
                emotion = smoothers[key].update(emotion)

            draw_box(frame, (x1, y1, x2, y2), emotion, emotion_conf, detect_conf)

            cv2.putText(
                frame, f"Face #{face_id}", (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

            face_results.append({
                "face_id": face_id,
                "box": (x1, y1, x2, y2),
                "emotion": emotion,
                "confidence": emotion_conf,
                "detect_conf": detect_conf,
                "prediction": prediction
            })

    return frame, face_results

# ============================================================
# MODE 1: PHOTO UPLOAD
# ============================================================

if mode == "📷 Upload Photo":

    st.title("📷 Upload a Photo")
    st.caption("Upload an image to detect faces and predict emotions.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Detecting faces & predicting emotions..."):
            annotated, face_results = process_frame(frame.copy(), yolo_conf)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="Prediction Result",
                use_container_width=True
            )

        with col2:
            if not face_results:
                st.warning("No faces detected. Try a clearer photo or lower the confidence threshold.")
            else:
                st.success(f"{len(face_results)} face(s) detected")

                for fr in face_results:
                    st.subheader(f"Face #{fr['face_id']}")
                    st.metric("Predicted Emotion", fr["emotion"], f"{fr['confidence']*100:.1f}% confidence")

                    top3_scores = top3(fr["prediction"])
                    chart_data = {emo: score for emo, score in top3_scores}
                    st.bar_chart(chart_data)

                    with st.expander("Full probability breakdown"):
                        for idx, emo in enumerate(EMOTIONS):
                            st.progress(
                                float(fr["prediction"][idx]),
                                text=f"{emo}: {fr['prediction'][idx]*100:.2f}%"
                            )
                    st.markdown("---")
    else:
        st.info("👆 Upload an image to get started.")

# ============================================================
# MODE 2: LIVE WEBCAM
# ============================================================

else:

    st.title("🎥 Live Webcam")
    st.caption("Real-time face & emotion detection from your webcam.")

    col1, col2 = st.columns([3, 1])

    with col2:
        run = st.checkbox("Start Camera")
        frame_placeholder_info = st.empty()

    with col1:
        frame_placeholder = st.empty()

    if run:

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            st.error("Cannot open webcam. Check camera permissions/connection.")
        else:
            fps_counter = FPS()
            smoothers = {}
            cached_result_frame = None
            frame_number = 0

            stop_placeholder = col2.empty()
            stop_button = stop_placeholder.button("Stop", key="stop_0")

            while run and not stop_button:

                success, frame = cap.read()

                if not success:
                    st.error("Failed to grab frame from webcam.")
                    break

                frame_number += 1

                # only run detection every FRAME_SKIP frames for speed,
                # reuse the last annotated frame otherwise
                if frame_number % FRAME_SKIP == 0 or cached_result_frame is None:
                    annotated, face_results = process_frame(
                        frame.copy(), yolo_conf, smoothers, "webcam_"
                    )
                    cached_result_frame = annotated
                else:
                    cached_result_frame = frame

                current_fps = fps_counter.update()
                draw_fps(cached_result_frame, current_fps)
                draw_title(cached_result_frame)

                frame_placeholder.image(
                    cv2.cvtColor(cached_result_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True
                )

                frame_placeholder_info.metric("FPS", current_fps)

                # small sleep to avoid pegging CPU / UI thread
                time.sleep(0.01)

                # re-check stop button state each loop (rendered into the
                # same placeholder so it doesn't stack new widgets)
                stop_button = stop_placeholder.button("Stop", key=f"stop_{frame_number}")

            cap.release()
    else:
        frame_placeholder.info("Tick 'Start Camera' to begin live detection.")