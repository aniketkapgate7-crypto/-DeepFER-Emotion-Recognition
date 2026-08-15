from pathlib import Path
import json
import time
import cv2
import numpy as np
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "deepfer_best.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"

# Emotion Color Mapping (BGR format)
EMOTION_COLORS = {
    "angry": (0, 0, 220),       # Red
    "disgust": (0, 140, 0),     # Dark Green
    "fear": (180, 0, 180),      # Purple
    "happy": (0, 220, 255),     # Yellow/Cyan
    "neutral": (200, 200, 200), # Light Gray
    "sad": (255, 100, 0),       # Blue
    "surprise": (0, 255, 120),  # Mint Green
}


def load_resources():
    m_path = MODEL_PATH if MODEL_PATH.exists() else PROJECT_ROOT / "models" / "deepfer_final.keras"
    if not m_path.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found at {CLASS_NAMES_PATH}")

    model = keras.models.load_model(m_path)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    return model, class_names


def preprocess_face(face_gray):
    resized = cv2.resize(face_gray, (48, 48))
    norm = resized.astype(np.float32)
    tensor = np.expand_dims(norm, axis=-1)
    tensor = np.expand_dims(tensor, axis=0)
    return tensor


def run_webcam_detection(camera_index=0, smooth_alpha=0.3):
    model, class_names = load_resources()

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if face_detector.empty():
        raise RuntimeError("Could not load OpenCV Haar Cascade face detector.")

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam camera device at index {camera_index}.")

    print("\nDeepFER Webcam Emotion Recognition Started.")
    print("  - Press 'q' to exit")
    print("  - Press 's' to save screenshot")

    screenshots_dir = PROJECT_ROOT / "results" / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    # Track smoothed probability per detected index
    smoothed_probs = None

    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to capture webcam frame.")
            break

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=6,
            minSize=(60, 60),
        )

        for (x, y, fw, fh) in faces:
            face_roi = gray[y:y + fh, x:x + fw]
            tensor = preprocess_face(face_roi)

            probs = model.predict(tensor, verbose=0)[0]

            if smoothed_probs is None:
                smoothed_probs = probs
            else:
                smoothed_probs = smooth_alpha * probs + (1 - smooth_alpha) * smoothed_probs

            top_idx = int(np.argmax(smoothed_probs))
            top_emotion = class_names[top_idx]
            top_conf = float(smoothed_probs[top_idx])

            color = EMOTION_COLORS.get(top_emotion, (0, 255, 0))

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

            # Label box
            label = f"{top_emotion.capitalize()} {top_conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, max(0, y - 30)), (x + tw + 10, y), color, cv2.FILLED)
            cv2.putText(
                frame,
                label,
                (x + 5, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

            # Draw mini probability bars on right side of bounding box
            bar_x = x + fw + 10
            bar_y = y
            bar_h = 14
            for idx, emo in enumerate(class_names):
                prob = float(smoothed_probs[idx])
                emo_color = EMOTION_COLORS.get(emo, (200, 200, 200))

                # Label text
                cv2.putText(
                    frame,
                    f"{emo[:3].capitalize()}",
                    (bar_x, bar_y + (idx * 16) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

                # Bar background
                cv2.rectangle(
                    frame,
                    (bar_x + 35, bar_y + (idx * 16) + 2),
                    (bar_x + 115, bar_y + (idx * 16) + 12),
                    (50, 50, 50),
                    cv2.FILLED,
                )

                # Filled bar
                bar_width = int(prob * 80)
                if bar_width > 0:
                    cv2.rectangle(
                        frame,
                        (bar_x + 35, bar_y + (idx * 16) + 2),
                        (bar_x + 35 + bar_width, bar_y + (idx * 16) + 12),
                        emo_color,
                        cv2.FILLED,
                    )

        # Header overlay
        cv2.rectangle(frame, (0, 0), (w, 35), (20, 20, 20), cv2.FILLED)
        cv2.putText(
            frame,
            "DeepFER Emotion Detection  |  Press 'q' to Quit  |  Press 's' for Screenshot",
            (15, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        cv2.imshow("DeepFER - Facial Emotion Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Exiting webcam detection.")
            break
        elif key == ord("s"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            ss_path = screenshots_dir / f"emotion_capture_{timestamp}.png"
            cv2.imwrite(str(ss_path), frame)
            print(f"Screenshot saved to {ss_path}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_detection()