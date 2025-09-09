# app.py — Bilingual (Arabic + English) Sentiment Analysis
# --------------------------------------------------------
# يدعم: نص واحد، CSV، PDF، DOCX + إدارة الموديلات + فحص البيئة

import os, sys, re, io, zipfile, json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# PDF & DOCX readers
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document
except Exception:
    Document = None

# ===== Lazy TensorFlow import =====
TF_IMPORT_ERROR = None
tf = None
tokenizer_from_json = None
pad_sequences = None

def ensure_tf():
    """Import TensorFlow only when needed."""
    global tf, tokenizer_from_json, pad_sequences, TF_IMPORT_ERROR
    if tf is not None:
        return True, None
    try:
        import tensorflow as _tf
        from tensorflow.keras.preprocessing.text import tokenizer_from_json as _tok_json
        from tensorflow.keras.preprocessing.sequence import pad_sequences as _pad
        tf = _tf
        tokenizer_from_json = _tok_json
        pad_sequences = _pad
        return True, None
    except Exception as e:
        TF_IMPORT_ERROR = e
        return False, str(e)

# ------------------------
# Config
# ------------------------
st.set_page_config(page_title="💬 Sentiment | تحليل المشاعر", page_icon="💬", layout="wide")
DEFAULT_MODEL_DIR = Path("bilingual_sentiment_model")
MAX_LEN = 96
CLASSES_FALLBACK = ["negative", "neutral", "positive"]

# ------------------------
# Lang utils
# ------------------------
ARABIC_RE = re.compile(r'[\u0600-\u06FF]')
AR_DIACRITICS = r"[\u0617-\u061A\u064B-\u0652\u0670]"

def detect_language_simple(text: str) -> str:
    return "ar" if ARABIC_RE.search(str(text)) else "en"

def ar_normalize(s: str) -> str:
    s = str(s)
    s = re.sub(AR_DIACRITICS, "", s)
    s = re.sub(r"[ـ]+", "", s)
    s = s.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    s = s.replace("ى","ي").replace("ؤ","و").replace("ئ","ي").replace("ة","ه")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def preprocess_text(txt: str, lang: str) -> str:
    return ar_normalize(txt) if lang == "ar" else txt

# ------------------------
# Arabic keyword override
# ------------------------
AR_NEG = {
    "حزين","زعلان","تعيس","سيئ","سيء","مكتئب","محبط","تعبان","كاره",
    "مزعج","رديء","سئ","كارثي","مقرف","فظيع","سيئة","زفت","غثيث",
    "مؤسف","مخيّب","أسوأ","ممل"
}
AR_POS = {
    "سعيد","مبسوط","فرحان","ممتاز","رائع","جميل","حلو","احب","أحب",
    "عجبني","مذهل","مسعد","هايل","كويس","ممتازه","تحفه","خيالي",
    "يفوز","حبيت","أفضل","مرضي","مبهر"
}

def override_ar_prediction(text: str, label: str, probs: np.ndarray, classes: List[str], margin: float = 0.15) -> str:
    if "neutral" not in classes:
        return label
    try:
        i_neu = classes.index("neutral")
        i_neg = classes.index("negative")
        i_pos = classes.index("positive")
    except ValueError:
        return label
    t = str(text)
    has_neg = any(w in t for w in AR_NEG)
    has_pos = any(w in t for w in AR_POS)
    if label == "neutral":
        if has_neg and (probs[i_neu] - probs[i_neg] <= margin):
            return "negative"
        if has_pos and (probs[i_neu] - probs[i_pos] <= margin):
            return "positive"
    return label

# ------------------------
# Loaders
# ------------------------
@st.cache_resource(show_spinner=False)
def load_lang_assets(model_root: Path, lang: str):
    ok, err = ensure_tf()
    if not ok:
        raise RuntimeError(f"TensorFlow import failed: {err}")

    lang_dir = Path(model_root) / lang
    if not lang_dir.exists():
        raise FileNotFoundError(f"Language folder not found: {lang_dir}")

    # tokenizer
    tok_path = lang_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"Missing tokenizer.json in {lang_dir}")
    with open(tok_path, "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())

    # label_map
    label_map_path = lang_dir / "label_map.json"
    if label_map_path.exists():
        try:
            with open(label_map_path, "r", encoding="utf-8") as f:
                classes = json.load(f)["classes"]
        except Exception:
            classes = CLASSES_FALLBACK
    else:
        classes = CLASSES_FALLBACK

    # model file
    candidates = [
        lang_dir / f"{lang}_best.keras",
        lang_dir / f"{lang}_final.keras",
        lang_dir / f"{lang}_best.h5",
        lang_dir / f"{lang}_final.h5",
        lang_dir / "saved_model",
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(f"No model file found in {lang_dir}")

    model = tf.keras.models.load_model(model_path)
    return tok, classes, model

def _predict_batch(texts: List[str], model_root: Path) -> pd.DataFrame:
    ok, err = ensure_tf()
    if not ok:
        raise RuntimeError(f"TensorFlow import failed: {err}")

    langs = ["ar" if ARABIC_RE.search(t or "") else "en" for t in texts]
    rows = []
    cache: Dict[str, Tuple[Any, Any, Any]] = {}

    for lang in ("ar", "en"):
        idxs = [i for i, l in enumerate(langs) if l == lang]
        if not idxs:
            continue
        if lang not in cache:
            tok, classes, model = load_lang_assets(model_root, lang)
            cache[lang] = (tok, classes, model)
        else:
            tok, classes, model = cache[lang]

        subset = [preprocess_text(texts[i], lang) for i in idxs]
        seq = tok.texts_to_sequences(subset)
        X = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        probs = model.predict(X, verbose=0)
        pred_idx = np.argmax(probs, axis=1)

        for j, i_global in enumerate(idxs):
            label = classes[int(pred_idx[j])] if int(pred_idx[j]) < len(classes) else str(int(pred_idx[j]))
            if lang == "ar":
                label = override_ar_prediction(texts[i_global], label, probs[j], classes)
            row = {
                "text": texts[i_global],
                "lang": lang,
                "label": label,
                "confidence": float(probs[j, pred_idx[j]]),
            }
            for ci, cname in enumerate(classes):
                row[f"p_{cname}"] = float(probs[j, ci])
            rows.append(row)
    return pd.DataFrame(rows)

# ------------------------
# Readers
# ------------------------
def read_pdf(file) -> List[str]:
    if PdfReader is None:
        raise RuntimeError("pypdf not installed")
    texts = []
    reader = PdfReader(file)
    for pg in reader.pages:
        t = (pg.extract_text() or "").strip()
        if t: texts.append(t)
    return texts

def read_docx(file) -> List[str]:
    if Document is None:
        raise RuntimeError("python-docx not installed")
    texts = []
    doc = Document(file)
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t: texts.append(t)
    return texts

def read_csv(file) -> pd.DataFrame:
    df = None
    for enc in ("utf-8","utf-8-sig","latin-1","cp1256"):
        try:
            file.seek(0); df = pd.read_csv(file, encoding=enc); break
        except UnicodeDecodeError:
            continue
    return df if df is not None else pd.read_csv(file)

# ------------------------
# Sidebar
# ------------------------
with st.sidebar:
    st.header("⚙️ Settings | الإعدادات")
    model_root = Path(st.text_input("Model directory | مسار الموديلات", value=str(DEFAULT_MODEL_DIR)))
    st.caption("bilingual_sentiment_model/ar & /en each: model + tokenizer.json + label_map.json")

# ------------------------
# Tabs
# ------------------------
st.title("💬 Sentiment Analysis | تحليل المشاعر (AR/EN)")
tabs = st.tabs([
    "📝 Single Text | نص واحد",
    "📎 File (CSV / PDF / DOCX) | ملف",
    "🧩 Model Manager | إدارة النماذج",
    "🩺 Environment | البيئة"
])

# ------------------------
# Tab 1 - Single text
# ------------------------
with tabs[0]:
    ok_tf, err_tf = ensure_tf()
    if not ok_tf:
        st.error("TensorFlow غير متاح. افتحي تبويب Environment لمشاهدة السبب.")
    else:
        t = st.text_area("Enter text (Arabic or English):", height=140,
                         placeholder="مثال: انا سعيد اليوم / I love this product")
        if st.button("Predict"):
            if t.strip():
                try:
                    df = _predict_batch([t], model_root)
                    row = df.iloc[0]
                    lang_badge = "🇸🇦 عربي" if row["lang"] == "ar" else "🇬🇧 English"
                    st.success(f"**Language:** {lang_badge}\n\n**Prediction:** {row['label']}  |  **Confidence:** {row['confidence']:.3f}")
                    prob_cols = [c for c in df.columns if c.startswith("p_")]
                    if prob_cols:
                        st.dataframe(df[prob_cols].T.rename(columns={0: "probability"}))
                except Exception as e:
                    st.error(e)

# ------------------------
# Tab 2 - File upload
# ------------------------
with tabs[1]:
    ok_tf, err_tf = ensure_tf()
    if not ok_tf:
        st.error("TensorFlow غير متاح.")
    else:
        up = st.file_uploader("Upload CSV / PDF / DOCX", type=["csv","pdf","docx"])
        if st.button("Run") and up:
            try:
                if up.name.endswith(".csv"):
                    df = read_csv(up)
                    if "text" not in df.columns:
                        df = df.rename(columns={df.columns[0]: "text"})
                    texts = df["text"].astype(str).tolist()
                elif up.name.endswith(".pdf"):
                    texts = read_pdf(up)
                else:
                    texts = read_docx(up)

                out_df = _predict_batch(texts, model_root)
                st.dataframe(out_df)
                st.download_button("Download CSV", data=out_df.to_csv(index=False), file_name="predictions.csv")
            except Exception as e:
                st.error(e)

# ------------------------
# Tab 3 - Model Manager
# ------------------------
with tabs[2]:
    st.info("ارفع ملفات الموديل (.keras/.h5) + tokenizer.json + label_map.json لكل لغة (ar/en).")

# ------------------------
# Tab 4 - Environment
# ------------------------
with tabs[3]:
    st.write("**Python:**", sys.version)
    ok_tf, err_tf = ensure_tf()
    st.write("**TensorFlow imported?**", ok_tf)
    if ok_tf:
        st.write("TF version:", tf.__version__)
    else:
        st.error(err_tf)
