import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import matplotlib.pyplot as plt
import numpy as np
import io
from detoxify import Detoxify
import nltk
import cv2
import imagehash
import os
import clip
import torchvision.transforms as transforms
import piexif

# Initial setup
st.set_page_config(page_title="VeritasAI - Image & Text Analyzer", layout="centered")
st.title("🤖 VeritasAI")
st.markdown("""
<small>
<p><strong>Designed & Created by Brandon Hicks</strong>  
<br>A.I. Ambassador | Ethical AI Developer</p>
</small>

---

📜 **Ethical Frameworks Applied**

- **Deontological Ethics** — rooted in principles of truth, transparency, and duty-based analysis.
- **Transparency & Explainability** — empowering users with meaningful, interpretable insights.

---
""", unsafe_allow_html=True)

# Load NLP models
@st.cache_resource
def load_nlp_models():
    sentiment_pipeline = pipeline("sentiment-analysis", framework="pt")
    detox_model = Detoxify('original')
    deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    deberta_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-small")
    simcse_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-base")
    simcse_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base")
    return sentiment_pipeline, detox_model, deberta_tokenizer, deberta_model, simcse_tokenizer, simcse_model

# Load CLIP
@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

# NLP utility
def get_sentence_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        output = model(**inputs, return_dict=True)
    return F.normalize(output.pooler_output, p=2, dim=1)

# Visualization - Toxicity Bar Chart
def plot_toxicity(scores):
    fig, ax = plt.subplots()
    labels = list(scores.keys())
    values = list(scores.values())
    ax.bar(labels, values, color='red')
    ax.set_title("Toxicity Scores")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Visualization - Sentiment Gauge
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
    return buf

# Image analysis utilities
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
    try:
        exif_dict = piexif.load(image.info.get("exif", b""))
        exif_clean = {}
        if "0th" in exif_dict:
            exif_clean["Make"] = exif_dict["0th"].get(piexif.ImageIFD.Make, b"").decode("utf-8", "ignore")
            exif_clean["Model"] = exif_dict["0th"].get(piexif.ImageIFD.Model, b"").decode("utf-8", "ignore")
        if "Exif" in exif_dict:
            exif_clean["DateTimeOriginal"] = exif_dict["Exif"].get(piexif.ExifIFD.DateTimeOriginal, b"").decode("utf-8", "ignore")
        if "GPS" in exif_dict:
            exif_clean["GPSInfo"] = exif_dict["GPS"]
        return exif_clean
    except Exception as e:
        return {"Error": f"Failed to extract EXIF data: {e}"}

def decode_gps(exif_gps):
    try:
        lat_ref = exif_gps.get(1).decode()
        lat = exif_gps.get(2)
        lon_ref = exif_gps.get(3).decode()
        lon = exif_gps.get(4)

        def convert(coord):
            d, m, s = coord
            return d[0]/d[1] + (m[0]/m[1])/60 + (s[0]/s[1])/3600

        latitude = convert(lat)
        longitude = convert(lon)
        if lat_ref == 'S':
            latitude *= -1
        if lon_ref == 'W':
            longitude *= -1

        return round(latitude, 6), round(longitude, 6)
    except Exception:
        return None, None

# Load models
sentiment_pipeline, detox_model, deberta_tokenizer, deberta_model, simcse_tokenizer, simcse_model = load_nlp_models()
clip_model, clip_preprocess, clip_device = load_clip()

# UI logic
mode = st.radio("Choose analysis mode:", ["Text", "Image"])

if mode == "Text":
    text_input = st.text_area("Enter text to analyze:")
    if st.button("Analyze Text") and text_input:
        sentiment = sentiment_pipeline(text_input)[0]
        st.markdown(f"**Sentiment:** `{sentiment['label']}` (Confidence: `{sentiment['score']:.2f}`)")

        gauge_image = plot_sentiment_gauge_dynamic(sentiment['score'], sentiment['label'])
        st.image(gauge_image, caption="Sentiment Gauge", use_container_width=True)

        toxicity = detox_model.predict(text_input)
        st.image(plot_toxicity(toxicity), caption="Toxicity Scores", use_container_width=True)

        neutral = "This is a neutral and fact-based version of the same headline."
        sim_score = F.cosine_similarity(
            get_sentence_embedding(text_input, simcse_tokenizer, simcse_model),
            get_sentence_embedding(neutral, simcse_tokenizer, simcse_model)
        ).item()
        st.markdown(f"📐 Similarity to neutral phrasing: `{sim_score:.2f}`")

elif mode == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        blur_flag = detect_blur_or_smoothness(image)
        label, conf = semantic_check_with_clip(image, clip_model, clip_preprocess, clip_device)

        st.markdown(f"🧠 CLIP Verdict: **{label}** (Confidence: `{conf:.2f}`)")
        st.markdown("⚠️ Image may be overly smooth." if blur_flag else "✅ No over-smoothing detected.")

        metadata = extract_exif_data(image)
        st.markdown("### 🧾 Metadata (EXIF)")
        for key, value in metadata.items():
            if key != "GPSInfo":
                st.markdown(f"**{key}**: {value}")

        if metadata.get("Make") and metadata.get("Model"):
            st.markdown("✅ Metadata suggests this was taken with a real camera.")
            metadata_confidence = 1
        elif metadata.get("Error") or not any(metadata.values()):
            st.markdown("⚠️ No metadata found — image may be edited, AI-generated, or stripped.")
            metadata_confidence = 0
        else:
            st.markdown("⚠️ Incomplete metadata — uncertain origin.")
            metadata_confidence = 0.5

        if "GPSInfo" in metadata:
            lat, lon = decode_gps(metadata["GPSInfo"])
            if lat and lon:
                st.markdown(f"📍 **Picture was likely taken here:** [View on Google Maps](https://www.google.com/maps?q={lat},{lon})")

        # Final Verdict Logic
        if label.lower().startswith("an ai") or conf < 0.5:
            if blur_flag:
                verdict = "🔴 Likely AI-Generated"
            else:
                verdict = "⚠️ Possibly Real – Smooth but semantically flagged"
        else:
            if metadata_confidence == 1:
                verdict = "✅ Likely Real – Verified by EXIF metadata"
            else:
                verdict = "⚠️ Suspected Fake — overly smooth" if blur_flag else "✅ Likely Real"

        st.markdown(f"### Final Verdict: {verdict}")
