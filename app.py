# app.py — Bilingual Sentiment (AR/EN) + Safe startup with full error display

import streamlit as st

def _render_startup_error(e):
    st.set_page_config(page_title="App error", page_icon="⚠️", layout="centered")
    st.title("⚠️ Startup error")
    st.write(
        "The app crashed during startup. See the full traceback below. "
        "Fix the top-most error and redeploy."
    )
    st.exception(e)

def run_app():
    # ---- ضع تطبيقك الحقيقي هنا (الكود الكامل الذي أعطيتك إياه سابقًا) ----
    # نصيحة: أبقي كل الاستيرادات الثقيلة (tensorflow) داخل دوال ensure_tf فقط.
    import os, sys, re, io, zipfile, json
    from pathlib import Path
    from typing import Dict, Any, Tuple, List
    import numpy as np
    import pandas as pd
    import streamlit as st

    try:
        from pypdf import PdfReader
        PDF_IMPORT_ERROR = None
    except Exception as _e_pdf:
        PdfReader = None
        PDF_IMPORT_ERROR = _e_pdf

    try:
        from docx import Document
        DOCX_IMPORT_ERROR = None
    except Exception as _e_docx:
        Document = None
        DOCX_IMPORT_ERROR = _e_docx

    TF_IMPORT_ERROR = None
    tf = None
    tokenizer_from_json = None
    pad_sequences = None

    def ensure_tf():
        nonlocal tf, tokenizer_from_json, pad_sequences, TF_IMPORT_ERROR
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

    st.set_page_config(page_title="💬 Sentiment | تحليل المشاعر", page_icon="💬", layout="wide")
    DEFAULT_MODEL_DIR = Path("bilingual_sentiment_model")
    MAX_LEN = 96
    CLASSES_FALLBACK = ["negative", "neutral", "positive"]

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

    AR_NEG = {
        "حزين","زعلان","تعيس","سيئ","سيء","مكتئب","محبط","تعبان","كاره","مزعج","رديء","سئ",
        "كارثي","مقرف","فظيع","سيئة","زفت","غثيث","مؤسف","مخيّب","أسوأ","أبداً ما عجبني","ممل"
    }
    AR_POS = {
        "سعيد","مبسوط","فرحان","ممتاز","رائع","جميل","حلو","احب","أحب","عجبني","مذهل",
        "مسعد","هايل","كويس","ممتازه","تحفه","خيالي","يفوز","حبيت","أفضل","مرضي","مبهر"
    }

    def override_ar_prediction(text: str, label: str, probs, classes, margin: float = 0.15) -> str:
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

    @st.cache_resource(show_spinner=False)
    def load_lang_assets(model_root: Path, lang: str):
        ok, err = ensure_tf()
        if not ok:
            raise RuntimeError(f"TensorFlow import failed: {err}")
        lang_dir = Path(model_root) / lang
        if not lang_dir.exists():
            raise FileNotFoundError(f"Language folder not found: {lang_dir}")

        tok_path = lang_dir / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"Missing tokenizer.json in {lang_dir}")
        with open(tok_path, "r", encoding="utf-8") as f:
            tok = tokenizer_from_json(f.read())

        label_map_path = lang_dir / "label_map.json"
        if label_map_path.exists():
            try:
                with open(label_map_path, "r", encoding="utf-8") as f:
                    classes = json.load(f)["classes"]
            except Exception:
                classes = CLASSES_FALLBACK
        else:
            classes = CLASSES_FALLBACK

        candidates = [
            lang_dir / f"{lang}_best.keras",
            lang_dir / f"{lang}_final.keras",
            lang_dir / f"{lang}_best.h5",
            lang_dir / f"{lang}_final.h5",
            lang_dir / "saved_model",
        ]
        model_path = next((p for p in candidates if p.exists()), None)
        if model_path is None:
            raise FileNotFoundError(
                f"No model file found in {lang_dir}. Expected one of: {', '.join(p.name for p in candidates)}"
            )
        model = tf.keras.models.load_model(model_path)
        return tok, classes, model

    def _predict_batch(texts: List[str], model_root: Path) -> pd.DataFrame:
        ok, err = ensure_tf()
        if not ok:
            raise RuntimeError(f"TensorFlow import failed: {err}")
        langs = ["ar" if ARABIC_RE.search(t or "") else "en" for t in texts]
        rows = []
        cache: Dict[str, Tuple[any, any, any]] = {}
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
            X = pad_sequences(seq, maxlen=96, padding="post", truncating="post")
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

    # ---------- UI ----------
    with st.sidebar:
        st.header("⚙️ Settings | الإعدادات")
        model_root = Path(st.text_input("Model directory | مسار الموديلات", value="bilingual_sentiment_model"))
        st.caption("ar/ و en/ لازم تحتوي: model (.keras أو .h5 أو saved_model) + tokenizer.json + label_map.json")

    st.title("💬 Sentiment Analysis | تحليل المشاعر (AR/EN)")
    tabs = st.tabs(["📝 Single", "📄 CSV/PDF/DOCX", "🧩 Models", "🩺 Env"])

    with tabs[0]:
        ok_tf, err_tf = ensure_tf()
        if not ok_tf:
            st.error("TensorFlow غير متوفر. شوفي تبويب Environment.")
        else:
            t = st.text_area("Enter text | أدخل نص:", height=140)
            if st.button("Predict | تنبؤ", type="primary"):
                if t.strip():
                    df = _predict_batch([t], model_root)
                    if df.empty:
                        st.warning("No output.")
                    else:
                        r = df.iloc[0]
                        lang_badge = "🇸🇦 عربي" if r["lang"] == "ar" else "🇬🇧 English"
                        st.success(f"Language: {lang_badge} | Label: `{r['label']}` | Confidence: `{r['confidence']:.3f}`")
                else:
                    st.warning("أدخل نص")

    with tabs[1]:
        ok_tf, err_tf = ensure_tf()
        if not ok_tf:
            st.error("TensorFlow غير متوفر. شوفي تبويب Environment.")
        else:
            up = st.file_uploader("Upload CSV/PDF/DOCX", type=["csv","pdf","docx"])
            if up and st.button("Run", type="primary"):
                try:
                    import pandas as pd
                    def read_csv(file):
                        df = None
                        for enc in ("utf-8","utf-8-sig","latin-1","cp1256"):
                            try:
                                file.seek(0); df = pd.read_csv(file, encoding=enc); break
                            except UnicodeDecodeError: continue
                        return df if df is not None else pd.read_csv(file)
                    texts = []
                    if up.name.lower().endswith(".csv"):
                        df = read_csv(up)
                        if "text" not in df.columns:
                            df = df.rename(columns={df.columns[0]: "text"})
                        texts = df["text"].astype(str).tolist()
                    elif up.name.lower().endswith(".pdf"):
                        if PdfReader is None: raise RuntimeError(f"pypdf not available: {PDF_IMPORT_ERROR}")
                        reader = PdfReader(up)
                        for pg in reader.pages:
                            t = (pg.extract_text() or "").strip()
                            if t: texts.append(t)
                    else:
                        if Document is None: raise RuntimeError(f"python-docx not available: {DOCX_IMPORT_ERROR}")
                        doc = Document(up)
                        for p in doc.paragraphs:
                            t = (p.text or "").strip()
                            if t: texts.append(t)
                    if not texts:
                        st.warning("No text found.")
                    else:
                        out_df = _predict_batch(texts, model_root)
                        st.dataframe(out_df, use_container_width=True)
                except Exception as e:
                    st.exception(e)

    with tabs[2]:
        st.caption("Upload model/tokenizer/label_map to ar/en.")
        st.info("استخدمي GitHub لرفع الملفات الكبيرة. Streamlit قد يرفض ملفات > 100MB.")
        st.code("""bilingual_sentiment_model/
  ├─ ar/ {ar_best.keras | ar_best.h5 | saved_model/} + tokenizer.json + label_map.json
  └─ en/ {en_best.keras | en_best.h5 | saved_model/} + tokenizer.json + label_map.json
""")

    with tabs[3]:
        import sys, os
        st.write("Python:", sys.version)
        st.write("CWD:", os.getcwd())
        st.write("pypdf:", PdfReader is not None)
        st.write("python-docx:", Document is not None)
        ok_tf, _ = ensure_tf()
        st.write("TF imported:", ok_tf)
        if ok_tf:
            st.write("TF version:", tf.__version__)
            st.write("GPUs:", len(tf.config.list_physical_devices('GPU')))
        st.write("Model root exists:", Path("bilingual_sentiment_model").exists())
        for lang in ("ar","en"):
            d = Path("bilingual_sentiment_model")/lang
            st.write(f"{lang} dir:", d.exists(), str(d))
            if d.exists():
                try:
                    st.code("\n".join([p.name for p in sorted(d.iterdir())]))
                except: pass

if __name__ == "__main__":
    try:
        run_app()
    except Exception as e:
        _render_startup_error(e)
