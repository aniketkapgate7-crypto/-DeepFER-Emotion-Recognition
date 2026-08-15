from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from PIL import Image
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "deepfer_best.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"

# Load OpenCV Haar Cascade Face Detector
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def load_resources(model_path=None, class_names_path=None):
    m_path = Path(model_path) if model_path else MODEL_PATH
    c_path = Path(class_names_path) if class_names_path else CLASS_NAMES_PATH

    if not m_path.exists():
        # Fallback to final model if best model doesn't exist
        fallback_path = m_path.parent / "deepfer_final.keras"
        if fallback_path.exists():
            m_path = fallback_path
        else:
            raise FileNotFoundError(f"Model file not found: {m_path}")

    if not c_path.exists():
        raise FileNotFoundError(f"Class-names file not found: {c_path}")

    model = keras.models.load_model(m_path)

    with open(c_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    return model, class_names


def preprocess_face_crop(face_img):
    """Resizes cropped face array (grayscale or BGR) to (48, 48, 1) float32 tensor."""
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img

    resized = cv2.resize(gray, (48, 48))
    normalized = resized.astype(np.float32)
    tensor = np.expand_dims(normalized, axis=-1)
    tensor = np.expand_dims(tensor, axis=0)
    return tensor


def detect_and_crop_faces(image_bgr):
    """Detects face regions using OpenCV Haar Cascade Classifier."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )
    return faces


def predict_image(image_input, model=None, class_names=None, detect_faces=True):
    """
    Predicts facial emotion for an input image (filepath, PIL Image, or numpy BGR array).
    Returns list of dicts: [{ 'bbox': (x,y,w,h), 'emotion': str, 'confidence': float, 'probabilities': dict }]
    """
    if model is None or class_names is None:
        model, class_names = load_resources()

    # Load image to BGR numpy array
    if isinstance(image_input, (str, Path)):
        img_bgr = cv2.imread(str(image_input))
        if img_bgr is None:
            raise ValueError(f"Could not load image from {image_input}")
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input
    else:
        raise TypeError("Unsupported image input type.")

    h, w = img_bgr.shape[:2]
    faces = detect_and_crop_faces(img_bgr) if detect_faces else []

    results = []

    if len(faces) == 0:
        # No face detected or detection disabled -> predict whole image
        tensor = preprocess_face_crop(img_bgr)
        probs = model.predict(tensor, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        prob_dict = {cls: float(probs[i]) for i, cls in enumerate(class_names)}

        results.append({
            "bbox": (0, 0, w, h),
            "emotion": class_names[top_idx],
            "confidence": float(probs[top_idx]),
            "probabilities": prob_dict
        })
    else:
        # Predict each detected face region
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        for (x, y, fw, fh) in faces:
            face_roi = gray[y:y + fh, x:x + fw]
            tensor = preprocess_face_crop(face_roi)
            probs = model.predict(tensor, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            prob_dict = {cls: float(probs[i]) for i, cls in enumerate(class_names)}

            results.append({
                "bbox": (int(x), int(y), int(fw), int(fh)),
                "emotion": class_names[top_idx],
                "confidence": float(probs[top_idx]),
                "probabilities": prob_dict
            })

    return results, img_bgr


def draw_predictions(img_bgr, predictions):
    """Draws bounding boxes and emotion labels on image."""
    annotated = img_bgr.copy()
    for res in predictions:
        x, y, w, h = res["bbox"]
        label = f"{res['emotion'].capitalize()}: {res['confidence']:.1%}"

        # Bounding Box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x, max(0, y - 25)), (x + tw, y), (0, 255, 0), cv2.FILLED)
        cv2.putText(
            annotated,
            label,
            (x, max(15, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Predict facial emotion from an image file")
    parser.add_argument("image_path", type=Path, help="Path to the target image file")
    parser.add_argument("--save-output", type=Path, default=None, help="Path to save annotated image")
    parser.add_argument("--no-face-detect", action="store_true", help="Disable OpenCV face detection and analyze whole image")

    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    predictions, img_bgr = predict_image(
        args.image_path,
        detect_faces=not args.no_face_detect
    )

    print(f"\nPredictions for {args.image_path.name}:")
    for i, pred in enumerate(predictions, 1):
        print(f"\n[Face #{i}] BBox: {pred['bbox']}")
        print(f"Primary Emotion : {pred['emotion']} ({pred['confidence']:.2%})")
        print("All Probabilities:")
        sorted_probs = sorted(pred["probabilities"].items(), key=lambda x: x[1], reverse=True)
        for emo, prob in sorted_probs:
            print(f"  - {emo:10s}: {prob:.2%}")

    if args.save_output:
        annotated_img = draw_predictions(img_bgr, predictions)
        cv2.imwrite(str(args.save_output), annotated_img)
        print(f"\nSaved annotated image to {args.save_output}")


if __name__ == "__main__":
    main()