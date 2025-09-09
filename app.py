import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Bilingual Sentiment (AR/EN)", page_icon="💬", layout="centered")

DEFAULT_MODEL_DIR = Path("bilingual_sentiment_model")
MAX_LEN = 96

ARABIC_RE = re.compile(r'[\u0600-\u06FF]')
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
    lang_dir = model_dir / lang
    with open(lang_dir / "tokenizer.json", "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())
    with open(lang_dir / "label_map.json", "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]
    model_path = lang_dir / f"{lang}_best.keras"
    if not model_path.exists():
        model_path = lang_dir / f"{lang}_best.h5"
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

st.title("💬 Bilingual Sentiment (Arabic + English)")

text = st.text_area("Enter text (Arabic or English):", height=120)
if st.button("Predict sentiment", type="primary"):
    if text.strip():
        try:
            res = predict_one(text, DEFAULT_MODEL_DIR)
            st.json(res)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter some text.")
