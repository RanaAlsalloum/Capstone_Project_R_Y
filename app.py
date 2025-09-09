# app.py
# ===============================
# Bilingual Sentiment Analysis App (AR/EN)
# With Debugging Logs
# ===============================

import os, json, re, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------- Streamlit Page Config ----------
st.set_page_config(page_title="Bilingual Sentiment (AR/EN)", page_icon="💬", layout="centered")

st.title("💬 Bilingual Sentiment Analysis (Arabic + English)")
st.write("This is a **debug version** – errors will be shown here instead of crashing silently.")

# ---------- Constants ----------
DEFAULT_MODEL_DIR = Path("bilingual_sentiment_model")
MAX_LEN = 96
ARABIC_RE = re.compile(r'[\u0600-\u06FF]')

# ---------- Helper Functions ----------
def detect_language_simple(text: str) -> str:
    return "ar" if ARABIC_RE.search(str(text)) else "en"

def ar_normalize(s: str) -> str:
    s = str(s)
    s = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670]", "", s)
    s = re.sub(r"[ـ]+", "", s)
    s = s.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    s = s.replace("ى","ي").replace("ؤ","و").replace("ئ","ي").replace("ة","ه")
    s = re.sub(r"\s+", " ", s).strip()
    return s

@st.cache_resource(show_spinner=False)
def load_lang_assets(model_dir: Path, lang: str):
    """Load tokenizer, label map, and model for a given language."""
    lang_dir = model_dir / lang
    st.write(f"🔍 Loading assets for: {lang} from {lang_dir.resolve()}")

    # Tokenizer
    tok_path = lang_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")
    with open(tok_path, "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())

    # Label Map
    lbl_path = lang_dir / "label_map.json"
    if not lbl_path.exists():
        raise FileNotFoundError(f"Label map not found: {lbl_path}")
    with open(lbl_path, "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    # Model
    model_path = lang_dir / f"{lang}_best.keras"
    if not model_path.exists():
        model_path = lang_dir / f"{lang}_best.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found for {lang}: expected *_best.keras or *_best.h5")

    model = tf.keras.models.load_model(model_path)
    return tok, classes, model

def predict_one(text: str, model_dir: Path):
    lang = detect_language_simple(text)
    txt = ar_normalize(text) if lang == "ar" else text
    tok, classes, model = load_lang_assets(model_dir, lang)
    seq = tok.texts_to_sequences([txt])
    X = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(X, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return {"lang": lang, "label": classes[idx], "confidence": float(probs[idx])}

# ---------- UI ----------
try:
    text = st.text_area("Enter text (Arabic or English):", height=120)

    if st.button("Predict sentiment", type="primary"):
        if text.strip():
            res = predict_one(text, DEFAULT_MODEL_DIR)
            st.success("✅ Prediction complete")
            st.json(res)
        else:
            st.warning("⚠️ Please enter some text.")

except Exception as e:
    st.error("🔥 An error occurred while running the app.")
    st.code(traceback.format_exc())
