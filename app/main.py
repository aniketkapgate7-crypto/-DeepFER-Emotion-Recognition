import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to sys.path to import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.predict_image import load_resources, predict_image, draw_predictions

# Page Configuration
st.set_page_config(
    page_title="DeepFER - Facial Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Dark Theme
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5, #06B6D4, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #9CA3AF;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1F2937;
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Emotion Color Mapping (Hex)
EMOTION_HEX = {
    "angry": "#EF4444",      # Red
    "disgust": "#10B981",    # Emerald Green
    "fear": "#A855F7",       # Purple
    "happy": "#F59E0B",      # Amber / Gold
    "neutral": "#6B7280",    # Gray
    "sad": "#3B82F6",        # Blue
    "surprise": "#EC4899",   # Pink
}


@st.cache_resource
def get_cached_model():
    """Loads and caches the model and class names."""
    try:
        model, class_names = load_resources()
        return model, class_names, None
    except Exception as e:
        return None, None, str(e)


def main():
    st.markdown('<div class="main-title">🎭 DeepFER Facial Emotion Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Powered Real-Time & Image-based Facial Expression Analysis</div>', unsafe_allow_html=True)

    model, class_names, err = get_cached_model()

    if err:
        st.error(f"⚠️ Model Loading Error: {err}")
        st.info("Please ensure that `models/deepfer_best.keras` exists or run `python src/train.py` first.")
        return

    # Sidebar Navigation
    st.sidebar.title("📌 Navigation")
    mode = st.sidebar.radio(
        "Choose Mode:",
        [
            "🖼️ Image Emotion Analyzer",
            "📹 Live Webcam Analysis",
            "📊 Analytics & Performance",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Model Info")
    st.sidebar.markdown(f"**Classes ({len(class_names)}):** {', '.join([c.capitalize() for c in class_names])}")
    st.sidebar.markdown("**Input Shape:** 48 × 48 Grayscale")
    st.sidebar.markdown("**Architecture:** 4-Block CNN + GAP")

    if mode == "🖼️ Image Emotion Analyzer":
        render_image_analyzer(model, class_names)
    elif mode == "📹 Live Webcam Analysis":
        render_webcam_analyzer(model, class_names)
    elif mode == "📊 Analytics & Performance":
        render_analytics_tab(class_names)


def render_image_analyzer(model, class_names):
    st.subheader("🖼️ Facial Image Emotion Analysis")
    st.write("Upload a facial photo or select sample images to predict emotion probabilities.")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear face photo for optimal detection.",
    )

    col1, col2 = st.columns([1, 1])

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)

        with col1:
            st.markdown("#### 📷 Original Image")
            st.image(pil_img, use_container_width=True)

        with st.spinner("Analyzing facial expressions..."):
            preds, img_bgr = predict_image(pil_img, model=model, class_names=class_names, detect_faces=True)
            annotated_bgr = draw_predictions(img_bgr, preds)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        with col2:
            st.markdown("#### 🎯 Emotion Detection Overlay")
            st.image(annotated_rgb, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Emotion Confidence Breakdown")

        for idx, res in enumerate(preds, 1):
            top_emotion = res["emotion"]
            top_conf = res["confidence"]
            probs = res["probabilities"]

            st.markdown(f"### Face #{idx} - Detected Emotion: **{top_emotion.upper()}** ({top_conf:.1%})")

            # Create Plotly Horizontal Bar Chart
            df_probs = pd.DataFrame(
                [
                    {"Emotion": emo.capitalize(), "Probability": prob * 100, "Color": EMOTION_HEX.get(emo, "#3B82F6")}
                    for emo, prob in probs.items()
                ]
            ).sort_values("Probability", ascending=True)

            fig = px.bar(
                df_probs,
                x="Probability",
                y="Emotion",
                orientation="h",
                text=df_probs["Probability"].apply(lambda x: f"{x:.1f}%"),
                color="Emotion",
                color_discrete_map={row["Emotion"]: row["Color"] for _, row in df_probs.iterrows()},
            )

            fig.update_layout(
                xaxis_title="Confidence Probability (%)",
                yaxis_title="",
                showlegend=False,
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(range=[0, 100]),
            )
            fig.update_traces(textposition="outside")

            st.plotly_chart(fig, use_container_width=True)


def render_webcam_analyzer(model, class_names):
    st.subheader("📹 Live Webcam Emotion Recognition")
    st.write("Capture a webcam snapshot below to perform real-time facial emotion recognition.")

    camera_image = st.camera_input("Take a photo with your webcam")

    if camera_image:
        pil_img = Image.open(camera_image)
        col1, col2 = st.columns(2)

        with st.spinner("Processing facial expressions..."):
            preds, img_bgr = predict_image(pil_img, model=model, class_names=class_names, detect_faces=True)
            annotated_bgr = draw_predictions(img_bgr, preds)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        with col1:
            st.markdown("#### Detected Faces & Overlay")
            st.image(annotated_rgb, use_container_width=True)

        with col2:
            st.markdown("#### Primary Emotion Breakdown")
            for idx, res in enumerate(preds, 1):
                emo = res["emotion"].capitalize()
                conf = res["confidence"]

                st.metric(
                    label=f"Face #{idx} Emotion",
                    value=emo,
                    delta=f"{conf:.1%} confidence",
                )

                df_probs = pd.DataFrame(
                    list(res["probabilities"].items()),
                    columns=["Emotion", "Probability"],
                )
                df_probs["Emotion"] = df_probs["Emotion"].str.capitalize()
                df_probs["Probability"] *= 100

                fig = px.pie(
                    df_probs,
                    values="Probability",
                    names="Emotion",
                    title="Emotion Probability Distribution",
                    hole=0.4,
                    color="Emotion",
                    color_discrete_map={e.capitalize(): EMOTION_HEX.get(e, "#6B7280") for e in class_names},
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)


def render_analytics_tab(class_names):
    st.subheader("📊 Model Analytics & Dataset Insights")

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Overall Test Accuracy", value="58.39%", delta="Baseline FER2013")
    col2.metric(label="Weighted F1-Score", value="57.57%")
    col3.metric(label="Target Classes", value=f"{len(class_names)}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Training History", "🧩 Confusion Matrix", "📋 Classification Report"])

    with tab1:
        hist_plot = PROJECT_ROOT / "results" / "training_history.png"
        if hist_plot.exists():
            st.image(str(hist_plot), caption="Training & Validation Accuracy/Loss Curves", use_container_width=True)
        else:
            st.info("Training history plot not found in `results/training_history.png`.")

    with tab2:
        cm_plot = PROJECT_ROOT / "results" / "confusion_matrix.png"
        if cm_plot.exists():
            st.image(str(cm_plot), caption="Test Set Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix plot not found in `results/confusion_matrix.png`.")

    with tab3:
        report_file = PROJECT_ROOT / "results" / "classification_report.txt"
        if report_file.exists():
            with open(report_file, "r", encoding="utf-8") as f:
                report_text = f.read()
            st.code(report_text, language="text")
        else:
            st.info("Classification report not found in `results/classification_report.txt`.")


if __name__ == "__main__":
    main()
