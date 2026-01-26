import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import matplotlib.pyplot as plt
import numpy as np
import io
import nltk
import cv2
import imagehash
import os
import torchvision.transforms as transforms
import piexif
import clip  # moved import here — safe now

# ────────────────────────────────────────────────
# MUST BE THE VERY FIRST STREAMLIT COMMAND
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="VeritasAI - Image & Text Analyzer",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="auto"
)

# ────────────────────────────────────────────────
# CACHED HEAVY MODEL LOADERS (use these instead of uncached ones)
# ────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Detoxify (toxicity model)...", ttl="2h")
def get_detoxify():
    from detoxify import Detoxify
    return Detoxify('original')

@st.cache_resource(show_spinner="Loading sentiment analysis...", ttl="2h")
def get_sentiment_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device="cpu"
    )

@st.cache_resource(show_spinner="Loading CLIP model...", ttl="2h")
def get_clip():
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

@st.cache_resource
def download_nltk_data():
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)

download_nltk_data()  # safe to call early

# ────────────────────────────────────────────────
# YOUR OTHER FUNCTIONS (moved up — define before UI)
# ────────────────────────────────────────────────

def get_sentence_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        output = model(**inputs, return_dict=True)
    return F.normalize(output.pooler_output, p=2, dim=1)

def plot_toxicity(scores):
    fig, ax = plt.subplots()
    labels = list(scores.keys())
    values = list(scores.values())
    ax.bar(labels, values, color='red')
    ax.set_title("Toxicity Scores")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)  # important: free memory
    return buf

def plot_sentiment_gauge_dynamic(score, sentiment_label):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_xticks([])
    theta = score * np.pi
    color = "green" if sentiment_label.upper() == "POSITIVE" else "red" if sentiment_label.upper() == "NEGATIVE" else "gray"
    angles = np.linspace(0, np.pi, 100)
    ax.plot(angles, np.full_like(angles, 1), lw=15, color="lightgray")
    ax.plot([theta], [1], marker='o', markersize=12, color=color)
    ax.plot([theta, theta], [0, 1], lw=2, color=color)
    ax.set_title(f"{sentiment_label.capitalize()} Sentiment ({score:.2f})", va='bottom', color=color, fontsize=12)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def detect_blur_or_smoothness(image):
    gray = cv2.cvtColor(np.array(image.convert("L")), cv2.COLOR_GRAY2BGR)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 100

def semantic_check_with_clip(image, model, preprocess, device):
    inputs = preprocess(image).unsqueeze(0).to(device)
    tokens = clip.tokenize(["a real photo", "an AI-generated image"]).to(device)
    with torch.no_grad():
        probs = model(inputs, tokens)[0].softmax(dim=-1).cpu().numpy()[0]
    labels = ["a real photo", "an AI-generated image"]
    return labels[np.argmax(probs)], max(probs)

def extract_exif_data(image):
    """
    Extracts basic EXIF metadata from a PIL Image.
    Returns a dict with Make, Model, DateTimeOriginal, and full GPSInfo (if present).
    """
    try:
        exif_dict = piexif.load(image.info.get("exif", b""))
        exif_clean = {}

        # Camera make & model
        if "0th" in exif_dict:
            exif_clean["Make"] = exif_dict["0th"].get(piexif.ImageIFD.Make, b"").decode("utf-8", "ignore").strip()
            exif_clean["Model"] = exif_dict["0th"].get(piexif.ImageIFD.Model, b"").decode("utf-8", "ignore").strip()

        # Original capture date/time
        if "Exif" in exif_dict:
            exif_clean["DateTimeOriginal"] = exif_dict["Exif"].get(piexif.ExifIFD.DateTimeOriginal, b"").decode("utf-8", "ignore").strip()

        # Keep full GPSInfo block for later decoding
        if "GPS" in exif_dict:
            exif_clean["GPSInfo"] = exif_dict["GPS"]

        return exif_clean

    except Exception as e:
        return {"Error": f"Failed to extract EXIF data: {str(e)}"}


def decode_gps(exif_gps):
    try:
        if not exif_gps:
            return None, None

        lat_ref = exif_gps.get(1)  # 'N' or 'S'
        lat = exif_gps.get(2)      # ((degrees, 1), (minutes, 1), (seconds, 1))
        lon_ref = exif_gps.get(3)  # 'E' or 'W'
        lon = exif_gps.get(4)

        if None in (lat_ref, lat, lon_ref, lon):
            return None, None

        def dms_to_decimal(dms):
            d, m, s = dms
            return d[0]/d[1] + (m[0]/m[1])/60 + (s[0]/s[1])/3600

        latitude = dms_to_decimal(lat)
        longitude = dms_to_decimal(lon)

        if lat_ref.decode("utf-8", "ignore") == 'S':
            latitude *= -1
        if lon_ref.decode("utf-8", "ignore") == 'W':
            longitude *= -1

        return round(latitude, 6), round(longitude, 6)

    except Exception:
        return None, None
# Display basic metadata
st.markdown("### 🧾 Metadata (EXIF)")
for key, value in metadata.items():
    if key != "GPSInfo" and key != "Error":
        st.markdown(f"**{key}**: {value}")
    elif key == "Error":
        st.error(value)

# GPS handling
gps_lat, gps_lon = decode_gps(metadata.get("GPSInfo"))
if gps_lat is not None and gps_lon is not None:
    maps_url = f"https://www.google.com/maps?q={gps_lat},{gps_lon}"
    st.markdown(
        f"📍 **Picture was likely taken here:** "
        f"[View on Google Maps]({maps_url})",
        unsafe_allow_html=True
    )
    st.caption(f"Coordinates: {gps_lat:.6f}, {gps_lon:.6f}")
else:
    st.markdown("📍 **No valid GPS coordinates found in metadata.**")
# ────────────────────────────────────────────────
# UI STARTS HERE
# ────────────────────────────────────────────────
st.title("🤖 VeritasAI")

st.markdown("""
### What VeritasAI Does
VeritasAI is an ethical AI-powered analyzer that helps you quickly evaluate text and images for authenticity, bias, and potential manipulation.

- **Text Analysis**: Detects sentiment, toxicity levels, and neutrality - with visual gauges and breakdowns to make insights clear and actionable.
- **Image Analysis**: Checks for signs of AI generation (via CLIP semantics + smoothness detection), extracts camera metadata (make/model/date), and shows GPS location (if embedded) on Google Maps.

All processing happens in your browser session - no data is stored or shared.

### How to Use It
1. Choose **Text** or **Image** mode using the toggle above.
2. For text: Paste or type content → click **Analyze Text** → review sentiment gauge, toxicity chart, and neutrality score.
3. For images: Upload a photo (JPG/PNG) → wait a moment → see authenticity verdict, metadata summary, GPS link (if available), and visual flags.

Upload responsibly - use only images/text you have rights to analyze.

---
Designed & Created by Brandon Hicks  
A.I. Ambassador | Ethical AI Developer
""", unsafe_allow_html=True)
mode = st.radio("Choose analysis mode:", ["Text", "Image"])

if mode == "Text":
    text_input = st.text_area("Enter text to analyze:")
    if st.button("Analyze Text") and text_input:
        with st.spinner("Analyzing..."):
            sentiment = get_sentiment_pipeline()(text_input)[0]
            st.markdown(f"**Sentiment:** `{sentiment['label']}` (Confidence: `{sentiment['score']:.2f}`)")
            gauge_image = plot_sentiment_gauge_dynamic(sentiment['score'], sentiment['label'])
            st.image(gauge_image, caption="Sentiment Gauge", use_container_width=True)

            toxicity = get_detoxify().predict(text_input)
            st.image(plot_toxicity(toxicity), caption="Toxicity Scores", use_container_width=True)

            neutral = "This is a neutral and fact-based version of the same headline."
            # If you still want SIMCSE — load it cached too (add function if needed)
            # For now commented to save memory
            # sim_score = ...
            # st.markdown(f"📐 Similarity to neutral phrasing: `{sim_score:.2f}`")

elif mode == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        with st.spinner("Processing image..."):
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            clip_model, clip_preprocess, clip_device = get_clip()
            label, conf = semantic_check_with_clip(image, clip_model, clip_preprocess, clip_device)
            st.markdown(f"🧠 CLIP Verdict: **{label}** (Confidence: `{conf:.2f}`)")

            blur_flag = detect_blur_or_smoothness(image)
            st.markdown("⚠️ Image may be overly smooth." if blur_flag else "✅ No over-smoothing detected.")

            metadata = extract_exif_data(image)
            st.markdown("### 🧾 Metadata (EXIF)")
            for key, value in metadata.items():
                if key != "GPSInfo":
                    st.markdown(f"**{key}**: {value}")

            metadata_confidence = 0
            if metadata.get("Make") and metadata.get("Model"):
                st.markdown("✅ Metadata suggests this was taken with a real camera.")
                metadata_confidence = 1
            elif metadata.get("Error") or not any(metadata.values()):
                st.markdown("⚠️ No metadata found — image may be edited, AI-generated, or stripped.")
            else:
                st.markdown("⚠️ Incomplete metadata — uncertain origin.")
                metadata_confidence = 0.5

            if "GPSInfo" in metadata:
                lat, lon = decode_gps(metadata["GPSInfo"])
                if lat and lon:
                    st.markdown(f"📍 **Picture was likely taken here:** [View on Google Maps](https://www.google.com/maps?q={lat},{lon})")

            # Final Verdict
            if label.lower().startswith("an ai") or conf < 0.5:
                verdict = "🔴 Likely AI-Generated" if blur_flag else "⚠️ Possibly Real – Smooth but semantically flagged"
            else:
                verdict = "✅ Likely Real – Verified by EXIF metadata" if metadata_confidence == 1 else "⚠️ Suspected Fake — overly smooth" if blur_flag else "✅ Likely Real"
            st.markdown(f"### Final Verdict: {verdict}")
