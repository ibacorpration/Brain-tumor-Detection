from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import io

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
)

CLASS_LABELS = ["pituitary", "glioma", "notumor", "meningioma"]
IMAGE_SIZE = 256

BASE_DIR = Path(__file__).resolve().parent

CANDIDATES = [
    BASE_DIR / "brain tumor_efficientnet_model.keras",
    BASE_DIR / "models" / "brain tumor_efficientnet_model.keras",
    BASE_DIR.parent / "brain tumor_efficientnet_model.keras",
    BASE_DIR.parent / "models" / "brain tumor_efficientnet_model.keras",
]

@st.cache_resource(show_spinner=False)
def get_model():
    for p in CANDIDATES:
        if p.exists():
            return load_model(str(p))

    raise FileNotFoundError(
        "Model file not found. Looked in:\n" +
        "\n".join([f"  - {p}" for p in CANDIDATES])
    )

def preprocess_image(uploaded_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return x, img

def predict_tumor(x: np.ndarray):
    model = get_model()
    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds, axis=1)[0])
    conf = float(np.max(preds, axis=1)[0])

    label = CLASS_LABELS[idx]
    if label == "notumor":
        display = "No Tumor"
    else:
        display = f"Tumor: {label}"

    return display, conf, preds[0]

# ----------------------------
# UI
# ----------------------------

st.title("🧠 Brain Tumor Detection")
st.caption("Upload an MRI image and the model will predict the tumor type (or no tumor).")

with st.expander("Model & preprocessing details", expanded=False):
    st.write(
        f"- Input size: **{IMAGE_SIZE}×{IMAGE_SIZE}**\n"
        "- RGB image, normalized to **[0..1]**\n"
        f"- Classes: **{', '.join(CLASS_LABELS)}**"
    )

uploaded = st.file_uploader(
    "Upload MRI image",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Choose an image file to get a prediction.")
    st.stop()

try:
    x, preview_img = preprocess_image(uploaded.getvalue())
except Exception as e:
    st.error(f"Couldn't read this file as an image.\n\nDetails: {e}")
    st.stop()

st.image(preview_img, caption="Uploaded image", use_container_width=True)

try:
    result, confidence, probs = predict_tumor(x)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Prediction failed.\n\nDetails: {e}")
    st.stop()

st.subheader("Result")
st.write(f"**{result}**")
st.write(f"Confidence: **{confidence * 100:.2f}%**")

st.subheader("Class probabilities")
prob_table = {label: float(p) for label, p in zip(CLASS_LABELS, probs)}
st.bar_chart(prob_table)

# Top-2 quick view
top2 = sorted(prob_table.items(), key=lambda kv: kv[1], reverse=True)[:2]
st.caption("Top predictions: " + " • ".join([f"{k}: {v*100:.2f}%" for k, v in top2]))

