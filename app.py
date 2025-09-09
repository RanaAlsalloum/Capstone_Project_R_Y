# app.py — minimal debug harness to locate the crash

import os, sys
import streamlit as st

st.set_page_config(page_title="Debug | تتبّع التشغيل", page_icon="🛠️", layout="centered")
st.title("🛠️ Debug startup | تتبّع تشغيل التطبيق")

try:
    st.write("✅ Step 1: imported streamlit OK")
    st.write("Python:", sys.version)
    st.write("CWD:", os.getcwd())

    # --- Check requirements presence ---
    problems = []
    try:
        import pandas as pd
        st.write("✅ pandas", pd.__version__)
    except Exception as e:
        problems.append(("pandas", e))
    try:
        from pypdf import PdfReader
        st.write("✅ pypdf OK")
    except Exception as e:
        problems.append(("pypdf", e))
    try:
        from docx import Document
        st.write("✅ python-docx OK")
    except Exception as e:
        problems.append(("python-docx", e))

    # Try tensorflow import, but catch error to see it on-screen
    tf_err = None
    try:
        import tensorflow as tf
        st.write("✅ tensorflow", tf.__version__)
    except Exception as e:
        tf_err = e
        st.error("❌ TensorFlow import failed")
        st.exception(e)

    # List model folders to ensure they exist
    import pathlib
    root = pathlib.Path("bilingual_sentiment_model")
    st.write("Model root exists:", root.exists(), str(root.resolve()))
    for lang in ("ar","en"):
        d = root/lang
        st.write(f"{lang} dir exists:", d.exists(), str(d))
        if d.exists():
            try:
                st.code("\n".join(p.name for p in sorted(d.iterdir())), language="bash")
            except:
                pass

    if problems:
        st.warning("Some libraries failed to import:")
        for name, err in problems:
            st.write(f"• {name}:"); st.exception(err)

    st.success("If you can see this, Streamlit is running. المشكلة فوق في الأخطاء المعروضة.")
except Exception as e:
    st.error("💥 Startup exception (outside app):")
    st.exception(e)
