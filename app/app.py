from pathlib import Path
import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "deepfer_best.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"


st.set_page_config(
    page_title="DeepFER",
    page_icon="😊",
    layout="wide"
)


@st.cache_resource
def load_resources():
    model = keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    if detector.empty():
        raise RuntimeError("Face detector could not be loaded.")

    return model, class_names, detector


def predict_emotion(face, model, class_names):
    grayscale_face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    resized_face = cv2.resize(grayscale_face, (48, 48))

    input_array = resized_face.astype(np.float32)
    input_array = np.expand_dims(input_array, axis=(0, -1))

    probabilities = model.predict(input_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    return (
        class_names[predicted_index],
        float(probabilities[predicted_index]),
        probabilities
    )


def analyse_image(image, model, class_names, detector):
    rgb_image = np.array(image.convert("RGB"))
    grayscale = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    faces = detector.detectMultiScale(
        grayscale,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )

    annotated_image = rgb_image.copy()
    results = []

    for x, y, width, height in faces:
        face = rgb_image[y:y + height, x:x + width]

        emotion, confidence, probabilities = predict_emotion(
            face,
            model,
            class_names
        )

        results.append({
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": probabilities
        })

        label = f"{emotion.capitalize()} {confidence:.1%}"

        cv2.rectangle(
            annotated_image,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            3
        )

        cv2.putText(
            annotated_image,
            label,
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return annotated_image, results


model, class_names, detector = load_resources()

st.title("DeepFER")
st.subheader("Facial Emotion Recognition Using Deep Learning")

st.info(
    "Upload a clear facial image or capture one using your camera."
)

input_method = st.radio(
    "Choose an input method",
    ["Upload image", "Use camera"],
    horizontal=True
)

selected_file = None

if input_method == "Upload image":
    selected_file = st.file_uploader(
        "Upload a facial image",
        type=["jpg", "jpeg", "png"]
    )
else:
    selected_file = st.camera_input("Capture an image")

if selected_file is not None:
    try:
        image = Image.open(selected_file).convert("RGB")
    except Exception:
        st.error(
            "This is not a valid image. Please select a real JPG, JPEG, "
            "or PNG file and avoid files beginning with '._'."
        )
        st.stop()

    annotated_image, results = analyse_image(
        image,
        model,
        class_names,
        detector
    )

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Detection result")
        st.image(annotated_image, width="stretch")

    with right_column:
        st.subheader("Prediction details")

        if not results:
            st.warning(
                "No face was detected. Try a clear, front-facing image."
            )

        for number, result in enumerate(results, start=1):
            st.markdown(f"### Face {number}")
            st.success(
                f"Prediction: {result['emotion'].capitalize()}"
            )
            st.metric(
                "Confidence",
                f"{result['confidence']:.1%}"
            )

            probability_data = {
                emotion.capitalize(): float(probability)
                for emotion, probability in zip(
                    class_names,
                    result["probabilities"]
                )
            }

            st.bar_chart(probability_data)

st.divider()

st.caption(
    "DeepFER estimates visible facial-expression categories. "
    "It does not determine a person's actual internal emotional state "
    "and should not be used for important decisions."
)
