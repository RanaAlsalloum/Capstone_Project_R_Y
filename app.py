# app.py — Bilingual Sentiment (Arabic + English)
# UI: EN+AR | Inputs: Single Text, CSV, PDF, DOCX, TXT | Model Manager
# Works with TF 2.20.0 (Python 3.12). Do NOT install tensorflow-macos/metal on Streamlit Cloud.

import json, re, io, os, zipfile
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# ML
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Files
from pypdf import PdfReader               # PDF
from docx import Document                 # DOCX

# ------------------------
# Page & Globals
# ------------------------
st.set_page_config(page_title="💬 Sentiment Analysis | تحليل المشاعر", page_icon="💬", layout="wide")

DEFAULT_MODEL_DIR = Path("bilingual_sentiment_model")
MAX_LEN = 96
CLASSES_FALLBACK = ["negative", "neutral", "positive"]  # if label_map.json missing

# Language detection & Arabic normalization
ARABIC_RE = re.compile(r'[\u0600-\u06FF]')
AR_DIACRITICS = r"[\u0617-\u061A\u064B-\u0652\u0670]"

def detect_language_simple(text: str) -> str:
    return "ar" if ARABIC_RE.search(str(text or "")) else "en"

def ar_normalize(s: str) -> str:
    s = str(s or "")
    s = re.sub(AR_DIACRITICS, "", s)
    s = re.sub(r"[ـ]+", "", s)
    s = s.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    s = s.replace("ى","ي").replace("ؤ","و").replace("ئ","ي").replace("ة","ه")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Arabic keyword overrides (expanded list)
AR_NEG = {
    "حزين","زعلان","تعيس","سيئ","سيء","مكتئب","محبط","تعبان","كاره","أكره","اكره","كرهت",
    "مزعج","سيئه","كارثي","سئ","أسوأ","اسوء","رديء","سيئة","سيئين","فضيع","فاشل","زفت","سخيف",
    "غضبان","غاضب","متضايق","مقرف","كارثة","糟", "سيّئ"
}
AR_POS = {
    "سعيد","مبسوط","فرحان","ممتاز","رائع","جميل","حلو","احب","أحب","عجبني","اعجبني",
    "ممتازه","ممتازة","مبهج","ايجابي","إيجابي","مسرور","ممتازين","خيال","أسطوري","نايس","تحفة","جميلة"
}

def override_ar_prediction(text: str, label: str, probs: np.ndarray, classes: List[str], margin: float = 0.18) -> str:
    """
    If model returns 'neutral', use simple keyword rules to flip to pos/neg when close.
    """
    if "neutral" not in classes or label != "neutral":
        return label
    ci = {c:i for i,c in enumerate(classes)}
    has_neg = any(w in text for w in AR_NEG)
    has_pos = any(w in text for w in AR_POS)
    p_neu = float(probs[ci["neutral"]])
    p_neg = float(probs[ci["negative"]]) if "negative" in ci else 0.0
    p_pos = float(probs[ci["positive"]]) if "positive" in ci else 0.0
    if has_neg and (p_neu - p_neg) <= margin:
        return "negative"
    if has_pos and (p_neu - p_pos) <= margin:
        return "positive"
    return label

# ------------------------
# Model / Tokenizer loading
# ------------------------
@st.cache_resource(show_spinner=False)
def load_lang_assets(model_root: Path, lang: str):
    """
    Load tokenizer, label_map, and model (.keras preferred, then .h5).
    We avoid loading SavedModel here because tf.saved_model.load() returns a
    serving function not a Keras model with .predict().
    """
    lang_dir = Path(model_root) / lang
    if not lang_dir.exists():
        raise FileNotFoundError(f"Missing folder: {lang_dir}")

    # tokenizer
    tok_path = lang_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"Missing tokenizer.json in {lang_dir}")
    with open(tok_path, "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())

    # classes
    lmap_path = lang_dir / "label_map.json"
    if lmap_path.exists():
        try:
            with open(lmap_path, "r", encoding="utf-8") as f:
                classes = json.load(f).get("classes", CLASSES_FALLBACK)
        except Exception:
            classes = CLASSES_FALLBACK
    else:
        classes = CLASSES_FALLBACK

    # model candidates
    candidates = [
        lang_dir / f"{lang}_best.keras",
        lang_dir / f"{lang}_final.keras",
        lang_dir / f"{lang}_best.h5",
        lang_dir / f"{lang}_final.h5",
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        # last resort: if user only has a SavedModel exported directory named 'saved_model'
        sm_dir = lang_dir / "saved_model"
        if sm_dir.exists():
            # Try to wrap SavedModel signature as a tf.function callable
            model = tf.saved_model.load(str(sm_dir))
            # Create a thin wrapper with a predict() compatible with our code
            def saved_predict(x):
                fn = model.signatures.get("serving_default") or next(iter(model.signatures.values()))
                out = fn(tf.convert_to_tensor(x))
                # take first tensor output
                first = list(out.values())[0]
                return first.numpy()
            return tok, classes, type("SavedModelWrapper",(object,),{"predict":lambda self, X, verbose=0: saved_predict(X)})()
        raise FileNotFoundError(
            f"No model file found in {lang_dir}. Expected one of: "
            f"{', '.join(p.name for p in candidates)} or saved_model/"
        )

    model = tf.keras.models.load_model(model_path)
    return tok, classes, model

def preprocess_text(txt: str, lang: str) -> str:
    return ar_normalize(txt) if lang == "ar" else (txt or "")

def predict_rows(texts: List[str], model_root: Path) -> pd.DataFrame:
    """
    Route each row to AR/EN model, apply override for Arabic, and return a dataframe of predictions.
    """
    langs = ["ar" if ARABIC_RE.search(t or "") else "en" for t in texts]
    rows = []
    cache: Dict[str, Tuple[Any, Any, Any]] = {}

    for lang in ("ar", "en"):
        idxs = [i for i, L in enumerate(langs) if L == lang]
        if not idxs:
            continue

        if lang not in cache:
            tok, classes, model = load_lang_assets(model_root, lang)
            cache[lang] = (tok, classes, model)
        else:
            tok, classes, model = cache[lang]

        subset = [preprocess_text(texts[i], lang) for i in idxs]
        seqs = tok.texts_to_sequences(subset)
        X = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
        probs = model.predict(X, verbose=0)
        preds = np.argmax(probs, axis=1)

        for j, gi in enumerate(idxs):
            label = classes[int(preds[j])] if int(preds[j]) < len(classes) else str(int(preds[j]))
            if lang == "ar":
                label = override_ar_prediction(subset[j], label, probs[j], classes)
            out = {
                "text": texts[gi],
                "lang": "Arabic" if lang == "ar" else "English",
                "label": label,
                "confidence": float(probs[j, preds[j]]),
            }
            for ci, cname in enumerate(classes):
                out[f"p_{cname}"] = float(probs[j, ci])
            rows.append(out)

    return pd.DataFrame(rows)

# ------------------------
# File readers (CSV / PDF / DOCX / TXT)
# ------------------------
def read_csv_any(up_file) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1256"):
        try:
            up_file.seek(0)
            return pd.read_csv(up_file, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    up_file.seek(0)
    return pd.read_csv(up_file)

def split_to_lines(text: str) -> List[str]:
    # split by newlines and sentence punctuation (., !, ?, ،)
    chunks = re.split(r"[\n\r]+|[\.!\?،]+", text or "")
    return [c.strip() for c in chunks if c and c.strip()]

def read_pdf(up_file) -> List[str]:
    reader = PdfReader(up_file)
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.extend(split_to_lines(t))
    return [t for t in texts if t]

def read_docx(up_file) -> List[str]:
    doc = Document(up_file)
    paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    # also split long paragraphs into sentences
    out = []
    for p in paras:
        out.extend(split_to_lines(p))
    return [t for t in out if t]

def read_txt(up_file) -> List[str]:
    data = up_file.read()
    # try encodings
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1256"):
        try:
            s = data.decode(enc)
            return split_to_lines(s)
        except Exception:
            continue
    return split_to_lines(data.decode("utf-8", errors="ignore"))

# ------------------------
# Model Manager (upload helper)
# ------------------------
def save_uploaded_model_files(lang: str, files: Dict[str, Any], root: Path) -> str:
    lang_dir = Path(root) / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    if files.get("zip"):
        try:
            with zipfile.ZipFile(files["zip"]) as z:
                z.extractall(lang_dir)
            return f"✅ Extracted ZIP into {lang_dir}"
        except Exception as e:
            return f"⚠️ Failed to extract ZIP: {e}"

    saved = []
    for key in ("model1", "model2"):
        f = files.get(key)
        if f and (f.name.endswith(".keras") or f.name.endswith(".h5")):
            out = lang_dir / f.name
            out.write_bytes(f.read())
            saved.append(out.name)
    tok = files.get("tokenizer")
    if tok:
        (lang_dir / "tokenizer.json").write_bytes(tok.read())
        saved.append("tokenizer.json")
    lmap = files.get("label_map")
    if lmap:
        (lang_dir / "label_map.json").write_bytes(lmap.read())
        saved.append("label_map.json")

    if not saved:
        return "⚠️ No files saved. Please upload model + tokenizer + label_map."
    return "✅ Saved: " + ", ".join(saved)

# ------------------------
# Sidebar
# ------------------------
with st.sidebar:
    st.header("⚙️ Settings | الإعدادات")
    model_root_str = st.text_input("Model folder | مجلد النماذج", value=str(DEFAULT_MODEL_DIR))
    model_root = Path(model_root_str)
    st.caption("Put two folders ar/ and en/ each with a model (.keras/.h5), tokenizer.json, label_map.json")

# ------------------------
# Main UI
# ------------------------
st.title("💬 Sentiment Analysis | تحليل المشاعر (AR/EN)")
tabs = st.tabs([
    "📝 Single Text | نص واحد",
    "📄 CSV / Files | ملفات CSV/نصية",
    "🧩 Model Manager | إدارة النماذج"
])

# TAB 1: Single Text
with tabs[0]:
    st.subheader("Single Text Prediction | التنبؤ لنص واحد")
    text = st.text_area(
        "Enter text (Arabic or English) | أدخل نص (عربي أو إنجليزي):",
        height=140,
        placeholder="مثال: المنتج رائع جدًا / This product is amazing!"
    )
    if st.button("Predict | تنبؤ", type="primary"):
        if text.strip():
            try:
                # route to model
                lang = detect_language_simple(text)
                tok, classes, model = load_lang_assets(model_root, lang)
                txt = preprocess_text(text, lang)
                seq = tok.texts_to_sequences([txt])
                X = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
                probs = model.predict(X, verbose=0)[0]
                idx = int(np.argmax(probs))
                label = classes[idx] if idx < len(classes) else str(idx)
                if lang == "ar":
                    label = override_ar_prediction(txt, label, probs, classes)
                conf = float(probs[idx])
                lang_badge = "🇸🇦 Arabic | عربي" if lang == "ar" else "🇬🇧 English | إنجليزي"
                st.success(f"**Language | اللغة:** {lang_badge}\n\n**Prediction | النتيجة:** `{label}`  |  **Confidence | الثقة:** `{conf:.3f}`")
                pd_df = pd.DataFrame({"class": classes, "probability": [float(p) for p in probs]})
                st.dataframe(pd_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error | خطأ: {e}")
        else:
            st.warning("Please enter some text | الرجاء إدخال نص.")

# TAB 2: CSV / Files
with tabs[1]:
    st.subheader("Batch Prediction | التنبؤ الدفعي")
    st.caption("Upload **CSV** with a 'text' column, or upload **PDF / DOCX / TXT**. | ارفع CSV يحوي عمود 'text' أو ملف PDF/DOCX/TXT.")
    colcsv, colfiles = st.columns(2)

    with colcsv:
        up_csv = st.file_uploader("Upload CSV | رفع ملف CSV", type=["csv"], key="csv_up")
        if up_csv and st.button("Run on CSV | تنفيذ على CSV", type="primary", key="run_csv"):
            try:
                df = read_csv_any(up_csv)
                if "text" not in df.columns:
                    # fallback to first column
                    first = df.columns[0]
                    df = df.rename(columns={first: "text"})
                texts = df["text"].astype(str).tolist()
                out_df = predict_rows(texts, model_root)
                result = pd.concat([df, out_df.drop(columns=["text"])], axis=1)
                st.success(f"Predicted {len(result)} rows | تم التنبؤ لعدد {len(result)} صفوف")
                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "Download results CSV | تحميل النتائج",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error | خطأ: {e}")

    with colfiles:
        up_misc = st.file_uploader("Upload PDF / DOCX / TXT | رفع PDF / DOCX / TXT", type=["pdf","docx","txt"], key="misc_up")
        if up_misc and st.button("Run on document | تنفيذ على المستند", type="primary", key="run_file"):
            try:
                ext = (up_misc.name or "").lower().split(".")[-1]
                if ext == "pdf":
                    texts = read_pdf(up_misc)
                elif ext == "docx":
                    texts = read_docx(up_misc)
                else:
                    texts = read_txt(up_misc)
                # filter out very short tokens
                texts = [t for t in texts if len(t.strip()) > 1]
                if not texts:
                    st.warning("No text found in file | لم يتم العثور على نص في الملف.")
                else:
                    out_df = predict_rows(texts, model_root)
                    st.success(f"Predicted {len(out_df)} segments | تم التنبؤ لعدد {len(out_df)} مقطع")
                    st.dataframe(out_df, use_container_width=True)
                    st.download_button(
                        "Download results CSV | تحميل النتائج",
                        data=out_df.to_csv(index=False).encode("utf-8"),
                        file_name="doc_predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error | خطأ: {e}")

# TAB 3: Model Manager
with tabs[2]:
    st.subheader("Model Manager | إدارة النماذج")
    st.caption("Upload model (.keras/.h5), tokenizer.json, label_map.json for each language. | ارفع ملفات الموديل لكل لغة.")
    col1, col2 = st.columns(2)

    for col, lang in zip((col1, col2), ("ar", "en")):
        with col:
            st.markdown(f"### {'Arabic 🇸🇦 | العربية' if lang=='ar' else 'English 🇬🇧 | الإنجليزية'}")
            up_zip = st.file_uploader(f"[{lang}] ZIP (optional) | ملف ZIP (اختياري)", type=["zip"], key=f"{lang}_zip")
            up_m1  = st.file_uploader(f"[{lang}] Model file 1 (.keras/.h5)", type=["keras","h5"], key=f"{lang}_m1")
            up_m2  = st.file_uploader(f"[{lang}] Model file 2 (.keras/.h5)", type=["keras","h5"], key=f"{lang}_m2")
            up_tok = st.file_uploader(f"[{lang}] tokenizer.json", type=["json"], key=f"{lang}_tok")
            up_map = st.file_uploader(f"[{lang}] label_map.json", type=["json"], key=f"{lang}_map")
            if st.button(f"Save {lang.upper()} files | حفظ ملفات {lang.upper()}", key=f"save_{lang}"):
                msg = save_uploaded_model_files(lang, {
                    "zip": up_zip, "model1": up_m1, "model2": up_m2,
                    "tokenizer": up_tok, "label_map": up_map
                }, model_root)
                st.info(msg)

    st.markdown("**Expected structure | هيكل المجلد المتوقع:**")
    st.code("""bilingual_sentiment_model/
  ├─ ar/
  │   ├─ ar_best.keras  (or ar_best.h5)
  │   ├─ tokenizer.json
  │   └─ label_map.json
  └─ en/
      ├─ en_best.keras  (or en_best.h5)
      ├─ tokenizer.json
      └─ label_map.json
""")
