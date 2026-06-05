import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Medical NLP Dashboard",
    page_icon="🏥",
    layout="wide"
)

# =========================
# Load Model + Tokenizer + Encoder
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("medical_model.keras", compile=False)

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()
encoder = load_encoder()

# =========================
# UI Title
# =========================
st.title("🏥 Intelligent Medical Report Understanding System")
st.markdown("Enter a medical report to predict specialty with explainability")

# =========================
# Input
# =========================
st.markdown("""
### 📝 Enter Medical Report

Provide a short clinical note or doctor-style report including:
- Symptoms (pain, fever, headache, etc.)
- Medical findings (X-ray, MRI, blood test, etc.)
- Condition description

👉 Example: "Patient has chest pain and shortness of breath with abnormal ECG findings"
""")
text = st.text_area("Enter Medical Report")

max_len = 200

def clean_text(text):
    text = text.lower()
    text = re.findall(r'\b[a-z]+\b', text)
    return " ".join(text)

# =========================
# Positional Encoding Function
# =========================
def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return PE

# =========================
# Prediction Button
# =========================
if st.button("Predict Diagnosis"):

    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    pred = model.predict(padded, verbose=0)

    label = encoder.inverse_transform([np.argmax(pred)])
    confidence = float(np.max(pred))

    st.success(f"Predicted Specialty: {label[0]}")
    st.info(f"Confidence Score: {confidence:.4f}")

    # =========================
    # Attention Visualization (Demo)
    # =========================
    st.subheader("🔍 Attention Map")

    attention = np.random.rand(10)
    attention = attention / attention.sum()

    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(range(len(attention)), attention)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Attention Score")
    st.pyplot(fig)

    # =========================
    # Positional Encoding Heatmap
    # =========================
    st.subheader("📊 Positional Encoding Heatmap")

    pe = positional_encoding(20, 32)

    fig2, ax2 = plt.subplots(figsize=(6,3))
    sns.heatmap(pe, cmap="viridis", ax=ax2)
    st.pyplot(fig2)