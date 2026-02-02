import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
import io
import nltk
import cv2
import os
import torchvision.transforms as transforms
import piexif
import clip
import warnings
import wikipedia
import re
from sentence_transformers import CrossEncoder
import imagehash

warnings.filterwarnings("ignore", category=UserWarning)

# ────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="VeritasAI - Image & Text Analyzer",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="auto"
)

# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### VeritasAI")
    st.markdown("**AI Image & Text Analyzer**")
    st.markdown("**Created by Brandon Hicks**  \nA.I. Developer")
    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("- Choose Text or Image mode")
    st.markdown("- Upload image or paste text")
    st.markdown("- Review results & signals")

# ────────────────────────────────────────────────
# CACHED MODELS
# ────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Detoxify...", ttl="2h")
def get_detoxify():
    from detoxify import Detoxify
    return Detoxify('original')

@st.cache_resource(show_spinner="Loading sentiment...", ttl="2h")
def get_sentiment():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device="cpu")

@st.cache_resource(show_spinner="Loading NLI fact-check...", ttl="2h")
def get_nli():
    return CrossEncoder('cross-encoder/nli-deberta-v3-base', device='cpu')

@st.cache_resource(show_spinner="Loading factuality model...", ttl="2h")
def get_factuality_model():
    return pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", device="cpu")

@st.cache_resource(show_spinner="Loading CLIP...", ttl="2h")
def get_clip():
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

@st.cache_resource(show_spinner="Loading AI Image Detector...", ttl="2h")
def get_ai_detector():
    return pipeline("image-classification", model="umm-maybe/AI-image-detector")

@st.cache_resource
def download_nltk():
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)

download_nltk()

# ────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────

def wikipedia_fact_check(claim: str, max_sentences: int = 5) -> tuple:
    claim = claim.strip().rstrip('.!?')
    if not claim:
        return "Unverified", "No claim provided.", 0.0
    try:
        wikipedia.set_lang("en")
        page = wikipedia.page(claim, auto_suggest=True, redirect=True)
        summary = wikipedia.summary(claim, sentences=max_sentences)

        # Use factuality model to determine stance
        fact_model = get_factuality_model()
        result = fact_model(summary, candidate_labels=["This confirms the claim", "This contradicts the claim", "This is neutral or unverified"])
        top_label = result['labels'][0]
        top_score = result['scores'][0]

        if top_label == "This confirms the claim":
            verdict = "True"
            explanation = f"Wikipedia confirms the claim: \"{summary[:250]}...\" (page: {page.title})"
        elif top_label == "This contradicts the claim":
            verdict = "False"
            explanation = f"Wikipedia indicates the claim is false: \"{summary[:250]}...\" (page: {page.title})"
        else:
            verdict = "Unverified"
            explanation = f"Wikipedia is neutral or unclear: \"{summary[:200]}...\" (page: {page.title})"

        return verdict, explanation, round(top_score, 2)
    except Exception as e:
        return "Unverified", f"Source lookup failed: {str(e)}", 0.0

def extract_exif_data(image):
    try:
        exif_dict = piexif.load(image.info.get("exif", b""))
        exif_clean = {}
        if "0th" in exif_dict:
            exif_clean["Make"] = exif_dict["0th"].get(piexif.ImageIFD.Make, b"").decode("utf-8", "ignore").strip()
            exif_clean["Model"] = exif_dict["0th"].get(piexif.ImageIFD.Model, b"").decode("utf-8", "ignore").strip()
        if "Exif" in exif_dict:
            exif_clean["DateTimeOriginal"] = exif_dict["Exif"].get(piexif.ExifIFD.DateTimeOriginal, b"").decode("utf-8", "ignore").strip()
        if "GPS" in exif_dict:
            exif_clean["GPSInfo"] = exif_dict["GPS"]
        return exif_clean
    except Exception:
        return {"Error": "Failed to extract EXIF"}

def decode_gps(exif_gps):
    try:
        if not exif_gps:
            return None, None
        lat_ref = exif_gps.get(1)
        lat = exif_gps.get(2)
        lon_ref = exif_gps.get(3)
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
    except:
        return None, None

def plot_toxicity(scores):
    fig, ax = plt.subplots()
    ax.bar(list(scores.keys()), list(scores.values()), color='red')
    ax.set_title("Toxicity Scores")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_sentiment_gauge(score, label):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_xticks([])
    theta = score * np.pi
    color = "green" if label.upper() == "POSITIVE" else "red" if label.upper() == "NEGATIVE" else "gray"
    angles = np.linspace(0, np.pi, 100)
    ax.plot(angles, np.full_like(angles, 1), lw=15, color="lightgray")
    ax.plot([theta], [1], marker='o', markersize=12, color=color)
    ax.plot([theta, theta], [0, 1], lw=2, color=color)
    ax.set_title(f"{label.capitalize()} ({score:.2f})", va='bottom', color=color, fontsize=12)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def detect_blur(image):
    gray = cv2.cvtColor(np.array(image.convert("L")), cv2.COLOR_GRAY2BGR)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 80

def detect_low_noise(image):
    gray = np.array(image.convert("L"))
    return np.std(gray) < 12

def error_level_analysis(image, quality=90):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)
    ela = Image.new("RGB", image.size)
    for x in range(image.width):
        for y in range(image.height):
            r, g, b = image.getpixel((x, y))
            rr, gg, bb = resaved.getpixel((x, y))
            ela.putpixel((x, y), (abs(r - rr)*2, abs(g - gg)*2, abs(b - bb)*2))
    ela_array = np.array(ela)
    return np.var(ela_array) < 50

def clip_check(image, model, preprocess, device):
    inputs = preprocess(image).unsqueeze(0).to(device)
    tokens = clip.tokenize(["a real photo", "an AI-generated image"]).to(device)
    with torch.no_grad():
        probs = model(inputs, tokens)[0].softmax(dim=-1).cpu().numpy()[0]
    labels = ["a real photo", "an AI-generated image"]
    return labels[np.argmax(probs)], max(probs)

# ────────────────────────────────────────────────
# MAIN UI
# ────────────────────────────────────────────────
st.title("🌐 VeritasAI 🔍")
st.markdown("Building intelligence in machines while searching for truth in life – Brandon Hicks")
st.caption("Upload an image to check for AI generation/manipulation or paste text to fact-check.")

mode = st.radio("Choose analysis mode:", ["Text", "Image"])

if mode == "Text":
    st.markdown("**Fact-Check Mode**: Paste a claim and optional evidence/context below. The app will tell you if the claim is true, false, or unverified.")

    with st.form(key="fact_check_form"):
        claim = st.text_input("Claim to verify:", placeholder="e.g., 'The Eiffel Tower is in Paris.'")
        
        evidence = st.text_area("Evidence or context (optional but improves accuracy):",
                                placeholder="e.g., 'The Eiffel Tower is a wrought-iron lattice tower built in 1889...'\n\nAdd any supporting or contradicting details here.",
                                height=150)
        
        analyze_button = st.form_submit_button("Analyze Claim", use_container_width=True, type="primary")

    if analyze_button and claim.strip():
        with st.spinner("Analyzing claim..."):
            if evidence.strip():
                nli_model = get_nli_fact_checker()
                scores = nli_model.predict([(claim, evidence)])
                probs = scores[0]
                label_idx = probs.argmax()
                label = ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"][label_idx]
                score = probs[label_idx]
                
                if label == "ENTAILMENT":
                    verdict = "True"
                    color = "green"
                    explanation = "The evidence supports the claim."
                elif label == "CONTRADICTION":
                    verdict = "False"
                    color = "red"
                    explanation = "The evidence contradicts the claim."
                else:
                    verdict = "Unverified"
                    color = "gray"
                    explanation = "The evidence is neutral or insufficient."
                
                st.markdown(f"### Verdict: <span style='color:{color}; font-weight:bold;'>{verdict}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence**: {score:.2%}")
                st.markdown(f"**Explanation**: {explanation}")
                st.caption(f"Claim: **{claim}**  \nEvidence: **{evidence}**")
            else:
                verdict, explanation, conf = wikipedia_fact_check(claim)
                color = {"True": "green", "False": "red", "Unverified": "gray"}.get(verdict, "gray")
                
                st.markdown(f"### Verdict: <span style='color:{color}; font-weight:bold;'>{verdict}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence**: {conf:.0%}")
                st.markdown(f"**Explanation**: {explanation}")
                st.caption(f"Source: Wikipedia  \nClaim: **{claim}**")
elif mode == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        with st.spinner("Processing image..."):
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Metadata
            metadata = extract_exif_data(image)
            st.markdown("### 🧾 Metadata (EXIF)")
            if "Error" in metadata:
                st.error(metadata["Error"])
            else:
                for key, value in metadata.items():
                    if key != "GPSInfo":
                        st.markdown(f"**{key}**: {value}")
                gps_lat, gps_lon = decode_gps(metadata.get("GPSInfo"))
                if gps_lat is not None and gps_lon is not None:
                    st.markdown(f"📍 **Likely taken here:** [Google Maps ↗](https://www.google.com/maps?q={gps_lat},{gps_lon}&z=16)")
                    st.caption(f"Coordinates: {gps_lat:.6f}°, {gps_lon:.6f}")
                else:
                    st.markdown("📍 **No valid GPS coordinates found.**")

            # AI Detection
            st.markdown("### 🔍 AI-Generation Analysis")
            detector = get_ai_detector()
            results = detector(image)
            ai_prob = 0.0
            label = "UNKNOWN"
            for r in results:
                lbl = r['label'].lower()
                if any(k in lbl for k in ['ai', 'generated', 'fake', 'synthetic']):
                    ai_prob = r['score']
                    label = "AI-Generated"
                    break
                elif any(k in lbl for k in ['real', 'human', 'authentic', 'photo']):
                    ai_prob = 1.0 - r['score']
                    label = "Real Photo"
                    break
            st.markdown(f"**AI Detector**: {label} (AI probability: **{ai_prob:.1%}**)")

            blur = detect_blur(image)
            low_noise = detect_low_noise(image)
            ela = error_level_analysis(image)

            st.markdown("⚠️ Overly smooth regions." if blur else "✅ No excessive smoothing.")
            st.markdown("⚠️ Suspiciously low pixel noise." if low_noise else "✅ Normal noise levels.")
            st.markdown("⚠️ Uniform compression artifacts." if ela else "✅ Varied compression artifacts.")

            # TinEye Reverse Search
            st.markdown("### 🔎 Reverse Image Search on TinEye")
            st.info("TinEye excels at exact matches and earliest appearances.")
            st.markdown("**Steps**: Right-click image above → 'Copy image address' or 'Open in new tab' → paste URL into [TinEye.com](https://tineye.com/)")
            st.caption("Real photos usually have older, diverse sources. AI images often lack history or appear suddenly.")

            # Verdict
            meta_conf = 1 if metadata.get("Make") and metadata.get("Model") else 0
            signals = sum([blur, low_noise, ela]) + (0.5 if meta_conf < 1 else 0)
            ai_signals = 0
            reasons = []
            if ai_prob > 0.70:
                ai_signals += 3.0
                reasons.append(f"Strong AI match ({ai_prob:.0%})")
            elif ai_prob > 0.45:
                ai_signals += 1.5
                reasons.append(f"Moderate AI suspicion ({ai_prob:.0%})")
            if blur:
                ai_signals += 0.8
                reasons.append("Over-smoothing")
            if low_noise:
                ai_signals += 0.8
                reasons.append("Low noise")
            if ela:
                ai_signals += 0.8
                reasons.append("Uniform ELA")
            if meta_conf < 1:
                ai_signals += 0.5
                reasons.append("Missing metadata")

            if ai_signals >= 5.0:
                verdict = "🔴 Very likely AI-generated"
                color = "red"
            elif ai_signals >= 3.5:
                verdict = "🟠 Probably AI or heavily edited"
                color = "orange"
            elif ai_signals >= 2.0:
                verdict = "🟡 Somewhat suspicious"
                color = "yellow"
            else:
                verdict = "✅ Most likely real photograph"
                color = "green"

            st.markdown(f"### Final Verdict: <span style='color:{color}; font-weight:bold;'>{verdict}</span>", unsafe_allow_html=True)
            st.markdown(f"**AI Signal Score**: {ai_signals:.1f}")
            if reasons:
                st.markdown("**Reasons:**")
                for r in reasons:
                    st.markdown(f"- {r}")
            else:
                st.markdown("No major suspicious signals.")
