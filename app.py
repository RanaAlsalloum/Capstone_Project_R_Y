# app.py — Bilingual (Arabic + English) Sentiment Analysis
# --------------------------------------------------------
# يدعم: نص واحد، CSV، PDF، DOCX + فحص البيئة
# مضاف: قواعد عربية لفك الحياد (نفي/تعجب/مكثّفات/إيموجي)

import os, sys, re, json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# ---- PDF & DOCX readers (اختياري) ----
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
st.caption("By: Rana Alsalloum, Yaqeen Adnan, Reem Al-Rshedi")
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
# Arabic rules/keywords to break neutrality
# ------------------------
# ------------------------
# Arabic rules/keywords to break neutrality  (STRONGER VERSION)
# ------------------------

# نستخدم قوائم "خام" ثم نطبّعها ar_normalize عشان نمسك كل أشكال الهمزات/التنوين/التاء المربوطة
AR_NEG_RAW = {
    "حزين","زعلان","تعيس","سيئ","سيء","سئ","مكتئب","محبط","تعبان","كاره",
    "مزعج","رديء","سيئة","كارثي","مقرف","فظيع","زفت","مخيّب","أسوأ","ممل",
    "كارثة","رداءة","غبن","قرف","ندمت","تافه","سيئين","بائس","يائس","منهك",
    "مقهور","محرج","خسارة","ضعيف","مأساة","مرعب","قاسي","حقير","مكسور",
    "دموع","كآبة","مزري","مؤلم","مفزع","حرام","مهزوم","مضايق","جرح","غبي",
    # سلبية مرّة (أقوى)
    "كارثي جدًا","مأساوي","فظيع جدًا","مقزز","مروع","كريه","بغيض",
    "مصيبة","فضيحة","مذل","مهين","مقرف جدًا","خيبة أمل","انكسار",
    "منهار","يائس جدًا","كارثة كبرى","أسوأ شيء","قمة السوء",
    # إضافات سلبية قويّة جدًا
    "كارثة لا تحتمل","كارثة لا تُحتمل","مأساة كبيرة","أمر فاضح","قمة الفشل","فشل ذريع",
    "مخيف جدًا","رهيب جدًا","بشع","قاسٍ جدًا","عار","مأزوم","محبط جدًا","مزري جدًا"
}

AR_POS_RAW = {
    "سعيد","مبسوط","فرحان","ممتاز","رائع","جميل","حلو","احب","أحب",
    "عجبني","مذهل","مسعد","هايل","كويس","ممتازه","تحفه","خيالي",
    "يفوز","حبيت","أفضل","مرضي","مبهر","روعة","يجنن","رهيب","مره حلو","فخم",
    "ممتن","شاكر","متحمس","متفائل","مُبتهج","سعادة","مُلهم","محظوظ","فرحة",
    "بهيج","منير","واو","برافو","نجاح","تفوق","فخور","مشرق","لطيف","محبوب",
    "إيجابي","قوي","مريح","مطمئن","مفرح","منجز","أحسنت","جميل جدًا","مُذهل",
    "أسطوري","مُفرح","سوبر","بديع","يبهج","يعجبني","فنان","قمة",
    # إيجابية مرّة (أقوى)
    "امتنان","ممتن جدًا","قمة السعادة","نجاح باهر","فائز","منتصر",
    "مبتهج للغاية","مبهج للغاية","رائع جدًا","جميل جدًا","سعيد جدًا",
    "طمأنينة","راحة","أمان","القادم أجمل","طاقة إيجابية","أحلامي تتحقق",
    "إلهام","فرحة كبيرة","إنجاز عظيم","إنجازات","لحظات جميلة",
    # إضافات إيجابية قويّة جدًا
    "ممتاز جدًا","روعة جدًا","مبهر جدًا","إنجاز ضخم","فخر كبير","فخور جدًا",
    "قمة الروعة","سعادة غامرة","تفوق كبير","مُلهم جدًا","مطمئن جدًا","سلام داخلي"
}

AR_NEGATIONS_RAW = {"مو","مش","ليس","ما","مو مره","مهو","مهوب","ولا","ابداً","أبدًا"}

AR_INTENSIFIERS_RAW = {
    "جداً","جدًا","مره","مرة","بشكل كبير","مرة كثير","قوي","للغاية","تمامًا",
    "بشدة","أوي","كتير","وايد","إلى حد بعيد","حقًا","بجد","مرة مره"
}

EMOJI_POS = {
    "😊","😍","🤩","😁","👍","💖","✨","👏","🥰","🌸","🌟","💐","🌷",
    "😎","🥳","🙌","💯","🔥","🌈","🤗","🎉","😇","💎"
}
EMOJI_NEG = {
    "😞","😡","🤬","😢","👎","💔","😠","😭","😓","😔","☹️","🙁","🥀",
    "🤕","🤢","😫","😩","😖","😣","😤"
}

# قوائم مطبّعة (حتى تمسك "ايجابيه/إيجابية/ايجابية..." إلخ)
AR_NEG = {ar_normalize(w) for w in AR_NEG_RAW}
AR_POS = {ar_normalize(w) for w in AR_POS_RAW}
AR_NEGATIONS = {ar_normalize(w) for w in AR_NEGATIONS_RAW}
AR_INTENSIFIERS = {ar_normalize(w) for w in AR_INTENSIFIERS_RAW}

# عبارات قويّة جدًا: ظهورها يقلب النتيجة مباشرة غالبًا
STRONG_POS = {ar_normalize(w) for w in {
    "امتنان","ممتن جدًا","طاقة إيجابية","القادم أجمل","سعيد جدًا","رائع جدًا",
    "نجاح باهر","إنجاز عظيم","إنجازات","لحظات جميلة","أحلامي تتحقق",
    "مطمئن","طمأنينة","راحة","أمان","مبتهج للغاية","مبهج للغاية","فرحة كبيرة",
    "إلهام","مُلهم","فخور جدًا","متفائل جدًا","قمة السعادة","سلام داخلي"
}}
STRONG_NEG = {ar_normalize(w) for w in {
    "كارثة كبرى","أسوأ شيء","مروع جدًا","مقزز جدًا","فضيحة","مذل","مهين",
    "كارثي جدًا","يائس جدًا","مصيبة كبيرة","قمة السوء","كارثة لا تُحتمل","فشل ذريع"
}}

# باراميترات أقوى لتدخل القواعد
EXCLAMATION_BOOST = 0.10
INTENSIFIER_BOOST = 0.12
RULE_CONF = 0.75
LOW_CONF = 0.90      # القاعدة تتدخل حتى لو ثقة النموذج 0.89
NEU_MARGIN = 0.35    # نفك الحياد أسرع باتجاه الأقرب

def _rule_score_ar(text: str) -> str | None:
    """قواعد سريعة: تُرجِع 'positive' أو 'negative' أو None."""
    t = ar_normalize(text)
    has_pos = any(w in t for w in AR_POS) or any(e in text for e in EMOJI_POS)
    has_neg = any(w in t for w in AR_NEG) or any(e in text for e in EMOJI_NEG)

    # نفي بسيط: "مو حلو" = سلبي ، "مو سيء" = أقرب لإيجابي
    negation = any(n in t for n in AR_NEGATIONS)
    if negation:
        if has_pos and not has_neg:
            has_pos, has_neg = False, True
        elif has_neg and not has_pos:
            has_pos, has_neg = True, False

    if has_pos and not has_neg:
        return "positive"
    if has_neg and not has_pos:
        return "negative"
    return None

def override_ar_prediction(
    text: str,
    label: str,
    probs: np.ndarray,
    classes: List[str],
    margin: float = NEU_MARGIN
) -> tuple[str, float]:
    """نسخة مُقوّاة: تسمح بقلب التنبؤ عند وجود مؤشرات عربية قويّة حتى لو ثقة الموديل مرتفعة نسبيًا."""
    # فهارس الفئات
    try:
        i_neg = classes.index("negative")
        i_neu = classes.index("neutral")
        i_pos = classes.index("positive")
    except ValueError:
        return label, float(np.max(probs))

    p_neg, p_neu, p_pos = float(probs[i_neg]), float(probs[i_neu]), float(probs[i_pos])
    top_p = max(p_neg, p_neu, p_pos)

    t = ar_normalize(text)

    # إشارات عامة وقويّة
    has_pos = any(w in t for w in AR_POS) or any(e in text for e in EMOJI_POS)
    has_neg = any(w in t for w in AR_NEG) or any(e in text for e in EMOJI_NEG)
    has_strong_pos = any(w in t for w in STRONG_POS)
    has_strong_neg = any(w in t for w in STRONG_NEG)

    # نفي
    negation = any(n in t for n in AR_NEGATIONS)
    if negation:
        if has_pos and not has_neg:
            has_pos, has_neg = False, True
        elif has_neg and not has_pos:
            has_pos, has_neg = True, False

    # (0) قلب مباشر لو فيه إشارة قويّة جدًا واضحة غير متعارضة
    if has_strong_pos and not has_strong_neg:
        return "positive", max(top_p, 0.85)
    if has_strong_neg and not has_strong_pos:
        return "negative", max(top_p, 0.85)

    # (1) لو محايد وقريب من طرف -> فك الحياد
    new_label = label
    if label == "neutral":
        if p_neu - p_neg <= margin:
            new_label = "negative"
        if p_neu - p_pos <= margin:
            new_label = "positive"

    # (2) لو ثقة الموديل أقل من LOW_CONF -> اسمح بالانقلاب حسب القاعدة
    if top_p < LOW_CONF:
        if has_pos and not has_neg:
            new_label = "positive"
            top_p = max(top_p, RULE_CONF)
        elif has_neg and not has_pos:
            new_label = "negative"
            top_p = max(top_p, RULE_CONF)

    # (3) تعزيزات التعجب/المكثّفات
    boost = 0.0
    excl = text.count("!")
    if excl >= 2: 
        boost += EXCLAMATION_BOOST
    if any(w in t for w in AR_INTENSIFIERS):
        boost += INTENSIFIER_BOOST

    if new_label == "positive" and boost > 0:
        top_p = min(0.99, max(top_p, RULE_CONF) + boost)
    elif new_label == "negative" and boost > 0 and excl >= 3:
        top_p = min(0.99, max(top_p, RULE_CONF) + boost/2)

    return new_label, top_p

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
            base_label = classes[int(pred_idx[j])] if int(pred_idx[j]) < len(classes) else str(int(pred_idx[j]))
            conf = float(probs[j, pred_idx[j]])
            label = base_label

            if lang == "ar":
                label, conf = override_ar_prediction(texts[i_global], base_label, probs[j], classes)

            row = {
                "text": texts[i_global],
                "lang": lang,
                "label": label,
                "confidence": float(conf),
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
                    st.success(f"**Language:** {lang_badge}\n\n**Prediction:** `{row['label']}`  |  **Confidence:** `{row['confidence']:.3f}`")
                    prob_cols = [c for c in df.columns if c.startswith("p_")]
                    if prob_cols:
                        st.markdown("**Probabilities:**")
                        st.dataframe(df[prob_cols].T.rename(columns={0: "probability"}))
                except Exception as e:
                    st.error(str(e))
            else:
                st.warning("اكتب نصًا أولاً.")

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
                if up.name.lower().endswith(".csv"):
                    df_in = read_csv(up)
                    if "text" not in df_in.columns:
                        df_in = df_in.rename(columns={df_in.columns[0]: "text"})
                    texts = df_in["text"].astype(str).tolist()
                elif up.name.lower().endswith(".pdf"):
                    texts = read_pdf(up)
                else:
                    texts = read_docx(up)

                out_df = _predict_batch(texts, model_root)
                st.dataframe(out_df, use_container_width=True)
                st.download_button("Download CSV", data=out_df.to_csv(index=False).encode("utf-8"),
                                   file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(str(e))

# ------------------------
# Tab 3 - Environment
# ------------------------
with tabs[2]:
    st.write("**Python:**", sys.version)
    ok_tf, err_tf = ensure_tf()
    st.write("**TensorFlow imported?**", ok_tf)
    if ok_tf:
        st.write("TF version:", tf.__version__)
        st.write("Num GPUs:", len(tf.config.list_physical_devices('GPU')))
    else:
        st.error(err_tf)
    st.write("**Model root exists?**", DEFAULT_MODEL_DIR.exists(), str(DEFAULT_MODEL_DIR.resolve()))
    for lang in ("ar","en"):
        d = DEFAULT_MODEL_DIR / lang
        st.write(f"**{lang} folder exists?**", d.exists(), str(d))
        if d.exists():
            try:
                st.code("\n".join([p.name for p in sorted(d.iterdir())]), language="bash")
            except:
                pass
