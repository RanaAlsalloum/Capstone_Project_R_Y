# app.py — Bilingual (Arabic + English) Sentiment | تحليل المشاعر
# يدعم: نص واحد، CSV، PDF، DOCX + إدارة الموديلات + فحص البيئة

import os, sys, re, io, zipfile, json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# مكتبات الملفات
from pypdf import PdfReader            # pip install pypdf
from docx import Document              # pip install python-docx

# ML
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------
# Page & Globals
# ------------------------
st.set_page_config(page_title="💬 Sentiment | تحليل المشاعر", page_icon="💬", layout="wide")
DEFAULT_MODEL_DIR = Path("bilingual_sentiment_model")
MAX_LEN = 96
CLASSES_FALLBACK = ["negative", "neutral", "positive"]  # لو label_map.json مفقود

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
# Arabic keyword override (يحرك الحالات المحايدة القريبة)
# ------------------------
AR_NEG = {
    "حزين","زعلان","تعيس","سيئ","سيء","مكتئب","محبط","تعبان","كاره","مزعج","رديء","سئ",
    "كارثي","مقرف","فظيع","سيئة","زفت","غثيث","مؤسف","مخيّب","أسوأ","أبداً ما عجبني","ممل"
}
AR_POS = {
    "سعيد","مبسوط","فرحان","ممتاز","رائع","جميل","حلو","احب","أحب","عجبني","مذهل",
    "مسعد","هايل","كويس","ممتازه","تحفه","خيالي","يفوز","حبيت","أفضل","مرضي","مبهر"
}

def override_ar_prediction(text: str, label: str, probs: np.ndarray, classes: List[str], margin: float = 0.15) -> str:
    # لا تعمل إلا إذا عندنا صنف "neutral"
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
        # قرّب القرار إن كان الفارق بسيط والعبارات المفتاحية موجودة
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
    """Load tokenizer, label_map, and model for a language ('ar'|'en')."""
    lang_dir = model_root / lang
    if not lang_dir.exists():
        raise FileNotFoundError(f"Language folder not found: {lang_dir}")

    # tokenizer
    tok_path = lang_dir / "tokenizer.json"
    if tok_path.exists():
        with open(tok_path, "r", encoding="utf-8") as f:
            tok = tokenizer_from_json(f.read())
    else:
        raise FileNotFoundError(f"Missing tokenizer.json in {lang_dir}")

    # classes
    label_map_path = lang_dir / "label_map.json"
    if label_map_path.exists():
        try:
            with open(label_map_path, "r", encoding="utf-8") as f:
                classes = json.load(f)["classes"]
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
        lang_dir / "saved_model",     # TF SavedModel directory
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            f"No model file found in {lang_dir}. "
            f"Expected one of: {', '.join(p.name for p in candidates)}"
        )

    model = tf.keras.models.load_model(model_path)
    return tok, classes, model


def _predict_batch(texts: List[str], model_root: Path) -> pd.DataFrame:
    """Predict for a list of raw texts (AR + EN mixed)."""
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
            # add per-class probs if available
            for ci, cname in enumerate(classes):
                row[f"p_{cname}"] = float(probs[j, ci])
            rows.append(row)

    return pd.DataFrame(rows)


def save_uploaded_model_files(lang: str, files: Dict[str, Any], root: Path) -> str:
    """Save uploaded files for a given language folder."""
    lang_dir = root / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    if files.get("zip"):
        try:
            with zipfile.ZipFile(files["zip"]) as z:
                z.extractall(lang_dir)
            return f"✅ Extracted into {lang_dir}"
        except Exception as e:
            return f"⚠️ ZIP extract failed: {e}"

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
# File readers
# ------------------------
def read_pdf(file) -> List[str]:
    texts = []
    try:
        reader = PdfReader(file)
        for pg in reader.pages:
            t = pg.extract_text() or ""
            t = t.strip()
            if t:
                texts.append(t)
    except Exception as e:
        st.error(f"PDF read error: {e}")
    return texts

def read_docx(file) -> List[str]:
    texts = []
    try:
        doc = Document(file)
        for p in doc.paragraphs:
            t = (p.text or "").strip()
            if t:
                texts.append(t)
    except Exception as e:
        st.error(f"DOCX read error: {e}")
    return texts

def read_csv(file) -> pd.DataFrame:
    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1256"):
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        file.seek(0)
        df = pd.read_csv(file)
    return df

# ------------------------
# Sidebar
# ------------------------
with st.sidebar:
    st.header("⚙️ Settings | الإعدادات")
    model_root = Path(st.text_input("Model directory | مسار الموديلات", value=str(DEFAULT_MODEL_DIR)))
    st.caption("Structure:\n"
               "bilingual_sentiment_model/ar (model + tokenizer.json + label_map.json)\n"
               "bilingual_sentiment_model/en (model + tokenizer.json + label_map.json)")

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

# ---- Tab 1: Single text
with tabs[0]:
    st.subheader("Single Text Prediction | التنبؤ لنص واحد")
    t = st.text_area("Enter text (Arabic or English) | أدخل نص (عربي أو إنجليزي):",
                     height=140, placeholder="مثال: انا سعيد اليوم / I love this product")
    if st.button("Predict | تنبؤ", type="primary"):
        if t.strip():
            try:
                df = _predict_batch([t], model_root)
                if df.empty:
                    st.warning("No output. Check models/files.")
                else:
                    row = df.iloc[0]
                    lang_badge = "🇸🇦 Arabic | عربي" if row["lang"] == "ar" else "🇬🇧 English | إنجليزي"
                    st.success(f"**Language | اللغة:** {lang_badge}\n\n"
                               f"**Prediction | النتيجة:** `{row['label']}`  |  "
                               f"**Confidence | الثقة:** `{row['confidence']:.3f}`")
                    # show probs table if present
                    prob_cols = [c for c in df.columns if c.startswith("p_")]
                    if prob_cols:
                        st.markdown("**Probabilities | الاحتمالات:**")
                        st.dataframe(df[prob_cols].T.rename(columns={0: "probability"}), use_container_width=True)
            except Exception as e:
                st.error("Error | خطأ:")
                st.exception(e)
        else:
            st.warning("Please enter some text | الرجاء إدخال نص.")

# ---- Tab 2: File (CSV / PDF / DOCX)
with tabs[1]:
    st.subheader("Batch Prediction from file | التنبؤ من ملف")
    up = st.file_uploader("Upload CSV (text column) or PDF or DOCX | ارفع CSV أو PDF أو DOCX",
                          type=["csv", "pdf", "docx"])
    run = st.button("Run | تشغيل", type="primary")
    if up and run:
        try:
            if up.name.lower().endswith(".csv"):
                df = read_csv(up)
                if "text" not in df.columns:
                    # اسم أول عمود يصبح text لو ما فيه عمود باسم text
                    df = df.rename(columns={df.columns[0]: "text"})
                texts = df["text"].astype(str).tolist()

            elif up.name.lower().endswith(".pdf"):
                texts = read_pdf(up)

            else:  # docx
                texts = read_docx(up)

            if not texts:
                st.warning("No text found | لم يتم العثور على نصوص.")
            else:
                out_df = _predict_batch(texts, model_root)
                if up.name.lower().endswith(".csv"):
                    # ألحق النتائج على CSV الأصلي
                    res = pd.concat([df.reset_index(drop=True), out_df.drop(columns=["text"]).reset_index(drop=True)], axis=1)
                else:
                    res = out_df

                st.success(f"Predicted {len(res)} rows | تم التنبؤ لعدد {len(res)} صفوف")
                st.dataframe(res, use_container_width=True)
                st.download_button("Download results CSV | تحميل النتائج",
                                   data=res.to_csv(index=False).encode("utf-8"),
                                   file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error("Error | خطأ:")
            st.exception(e)

# ---- Tab 3: Model Manager
with tabs[2]:
    st.subheader("Model Manager | إدارة النماذج")
    st.caption("Upload model (.keras/.h5), tokenizer.json, and label_map.json for Arabic and English.\n"
               "ارفع ملفات الموديل (.keras/.h5) و tokenizer.json و label_map.json لكل لغة.")
    col1, col2 = st.columns(2)
    for col, lang_label in zip((col1, col2), ("ar", "en")):
        with col:
            st.markdown(f"### {'Arabic 🇸🇦 | العربية' if lang_label=='ar' else 'English 🇬🇧 | الإنجليزية'}")
            up_zip = st.file_uploader(f"[{lang_label}] ZIP (optional) | ملف ZIP (اختياري)", type=["zip"], key=f"{lang_label}_zip")
            up_model1 = st.file_uploader(f"[{lang_label}] Model file 1 | ملف الموديل 1 (.keras/.h5)", type=["keras","h5"], key=f"{lang_label}_m1")
            up_model2 = st.file_uploader(f"[{lang_label}] Model file 2 | ملف الموديل 2 (.keras/.h5)", type=["keras","h5"], key=f"{lang_label}_m2")
            up_tok   = st.file_uploader(f"[{lang_label}] tokenizer.json", type=["json"], key=f"{lang_label}_tok")
            up_map   = st.file_uploader(f"[{lang_label}] label_map.json", type=["json"], key=f"{lang_label}_map")
            if st.button(f"Save {lang_label.upper()} files | حفظ ملفات {lang_label.upper()}", key=f"save_{lang_label}"):
                files = {"zip": up_zip, "model1": up_model1, "model2": up_model2, "tokenizer": up_tok, "label_map": up_map}
                msg = save_uploaded_model_files(lang_label, files, model_root)
                st.info(msg)

    st.markdown("**Expected folder structure | هيكل المجلد المتوقع:**")
    st.code("""bilingual_sentiment_model/
  ├─ ar/
  │   ├─ ar_best.keras / ar_best.h5 / saved_model/
  │   ├─ tokenizer.json
  │   └─ label_map.json
  └─ en/
      ├─ en_best.keras / en_best.h5 / saved_model/
      ├─ tokenizer.json
      └─ label_map.json
""")

# ---- Tab 4: Environment / Health
with tabs[3]:
    st.subheader("Environment check | فحص البيئة")
    st.write("**Python:**", sys.version)
    try:
        st.write("**TensorFlow:**", tf.__version__)
        st.write("**Num GPUs:**", len(tf.config.list_physical_devices('GPU')))
    except Exception as e:
        st.error("TensorFlow import failed | فشل استيراد TensorFlow")
        st.exception(e)

    try:
        import numpy as np; import pandas as pd; import h5py
        st.write("**NumPy:**", np.__version__)
        st.write("**Pandas:**", pd.__version__)
        st.write("**h5py:**", h5py.__version__)
    except Exception as e:
        st.error("Base libs import failed")
        st.exception(e)

    st.write("**Model root exists?**", DEFAULT_MODEL_DIR.exists(), str(DEFAULT_MODEL_DIR.resolve()))
    for lang in ("ar","en"):
        d = DEFAULT_MODEL_DIR / lang
        st.write(f"**{lang} dir exists?**", d.exists(), str(d))
        if d.exists():
            try:
                st.code("\n".join([p.name for p in sorted(d.iterdir())]), language="bash")
            except Exception:
                pass
