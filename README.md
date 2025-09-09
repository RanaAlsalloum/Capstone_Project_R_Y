# 💬 Bilingual Sentiment Analysis App (Arabic + English)

This web application can classify **Arabic** and **English** text into three sentiment categories:  
- Positive 😀  
- Neutral 😐  
- Negative 😞  

It supports both **single text prediction** and **batch prediction** from files (CSV, PDF, DOCX).  
The app also includes an **Environment Check** tab to help debug dependencies.  

🔗 **Live Demo on Streamlit Cloud:**  
👉 [Open the App](https://capstoneprojectry-dstete5v7yfp2crbgrlblf.streamlit.app/)

---

## ✨ Features
- Auto-detects language (**Arabic/English**)  
- Supports input as:
  - Single text box  
  - CSV (with text column)  
  - PDF / DOCX documents  
- Handles preprocessing for Arabic text (normalization, diacritics removal)  
- Includes **rule-based overrides** for Arabic (negation, intensifiers, emojis) to reduce "neutral" bias  
- Model manager: upload `.keras` / `.h5` models, `tokenizer.json`, and `label_map.json`  
- Environment tab: check Python, TensorFlow, and GPU availability  

---

## 🏗 Project Structure
```
bilingual_sentiment_model/
  ├─ ar/
  │   ├─ ar_best.keras / ar_best.h5
  │   ├─ tokenizer.json
  │   └─ label_map.json
  └─ en/
      ├─ en_best.keras / en_best.h5
      ├─ tokenizer.json
      └─ label_map.json
app.py
requirements.txt
README.md
```

---

## ⚙️ Installation & Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/USERNAME/Capstone_Project_R_Y.git
   cd Capstone_Project_R_Y
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 🚀 Deploy on Streamlit Cloud
1. Push this repo to GitHub.  
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).  
3. Connect your GitHub account and select this repo.  
4. Set **`app.py`** as the main entry file.  
5. Done! Your app will be live 🎉  

---

## 📊 Example Usage

### Single Text Input
- **Input:**  
  > انا سعيد جدًا اليوم  

- **Output:**  
  - Language: Arabic 🇸🇦  
  - Sentiment: Positive 😀  
  - Confidence: 0.92  

---

## 🔧 Requirements
Key dependencies:
- `streamlit==1.38.0`
- `tensorflow==2.15.0` (or compatible with Python 3.11+)
- `numpy`, `pandas`
- `pypdf`, `python-docx`

---

## 👩‍💻 Team Members
This project was developed as part of the **AI Practitioner Diploma – Capstone Project (2025)**:  

- **Rana Alsalloum**  
- **Yaqeen Adnan**  
- **Reem Al-Rshedi**  

📍 National Information Technology Academy (NITA), Saudi Arabia  
