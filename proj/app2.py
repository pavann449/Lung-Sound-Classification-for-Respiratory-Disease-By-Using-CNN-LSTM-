# app.py — Lung Sound Analyzer (Sky-Blue Theme + Audio Recording)

import os
import io
import time
import base64
import tempfile
from datetime import datetime

import streamlit as st
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import qrcode

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "best_model.pth"
CLASSES_PATH = "classes.npy"

SAMPLE_RATE = 22050
DURATION = 5.0
N_MELS = 128

st.set_page_config(
    page_title="Lung Sound — Sky Blue",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# SKY-BLUE THEME
# ==========================
def inject_css():
    st.markdown("""
    <style>
    html, body, [class*="css"], label, div, span, p, h1, h2, h3 {
        color: #0b2239 !important;
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(180deg,#e3f5ff,#cbecff,#b6e2ff,#a8dcff);
        min-height: 100vh;
    }
    .header {
        padding: 20px;
        border-radius: 16px;
        background: rgba(255,255,255,0.4);
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
        box-shadow: 0 6px 22px rgba(0,0,0,0.15);
    }
    .logo {
        width: 58px; height: 58px;
        border-radius: 14px;
        background: linear-gradient(135deg,#0284c7,#0ea5e9);
        display:flex; align-items:center; justify-content:center;
        font-size:22px; font-weight:800; color:white;
    }
    .card {
        background: rgba(255,255,255,0.55);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    .card-strong { border-left: 6px solid #0ea5e9; }
    .section-title { font-size: 16px; font-weight: 700; margin-bottom: 12px; }
    .stButton>button {
        background: linear-gradient(90deg,#0ea5e9,#0284c7);
        color: #fff;
        border-radius: 12px;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ==========================
# HEADER
# ==========================
st.markdown("""
<div class="header">
    <div class="logo">LS</div>
    <div>
        <h2>Lung Sound Analysis</h2>
        <p>Sky-Blue Medical Theme</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================
# MODEL
# ==========================
class CNNClassifier(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*16*27,128)
        self.fc2 = nn.Linear(128,n)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

@st.cache_resource
def load_model():
    classes = np.load(CLASSES_PATH, allow_pickle=True).tolist()
    model = CNNClassifier(len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, classes

# ==========================
# AUDIO PROCESSING
# ==========================
def extract_features(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    y = y[:int(sr*DURATION)]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    log = librosa.power_to_db(mel, ref=np.max)
    log = (log - log.mean()) / (log.std() + 1e-9)
    return log, y, sr

def mel_to_image(log):
    img = (log - log.min()) / (log.max() + 1e-9)
    return (img * 255).astype(np.uint8)

def waveform_image(sig, sr):
    fig, ax = plt.subplots(figsize=(6,1.5))
    ax.plot(np.linspace(0,len(sig)/sr,len(sig)), sig, color="#0284c7")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ==========================
# PDF REPORT
# ==========================
def make_pdf(audio_name, disease, conf, mel, name, age, notes):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, h-60, "Lung Sound Analysis Report")

    c.setFont("Helvetica", 11)
    c.drawString(40, h-100, f"Patient Name: {name}")
    c.drawString(40, h-120, f"Age: {age}")
    c.drawString(40, h-140, f"Notes: {notes}")
    c.drawString(40, h-160, f"Audio File: {audio_name}")
    c.drawString(40, h-180, f"Diagnosis: {disease}")
    c.drawString(40, h-200, f"Confidence: {conf:.2f}%")

    img = Image.fromarray(mel_to_image(mel))
    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    img_buf.seek(0)

    c.drawImage(ImageReader(img_buf), 40, h-450, width=400, height=220)

    qr = qrcode.make(f"{name}|{disease}|{conf:.2f}")
    qr_buf = io.BytesIO()
    qr.save(qr_buf, format="PNG")
    qr_buf.seek(0)
    c.drawImage(ImageReader(qr_buf), 40, h-650, width=110, height=110)

    c.save()
    buf.seek(0)
    return buf

# ==========================
# UI
# ==========================
left, right = st.columns([1.6, 1])

with left:
    st.markdown('<div class="card card-strong">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload or Record Audio</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload lung sound", type=["wav", "mp3"])
    recorded = st.audio_input("Or record using microphone")
    st.caption("Record ~5 seconds while patient breathes normally")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    name = st.text_input("Patient Name")
    age = st.text_input("Age")
    notes = st.text_area("Clinical Notes")
    analyze = st.button("Analyze")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    wf_placeholder = st.empty()
    spec_placeholder = st.empty()
    result_placeholder = st.empty()

# ==========================
# ANALYSIS
# ==========================
if analyze:
    if not uploaded and not recorded:
        st.error("Please upload or record an audio file.")
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

        if uploaded:
            tmp.write(uploaded.read())
            audio_name = uploaded.name
        else:
            tmp.write(recorded.getvalue())
            audio_name = f"recorded_{int(time.time())}.wav"

        tmp.flush()

        st.audio(tmp.name)

        mel, sig, sr = extract_features(tmp.name)

        wf_placeholder.image(waveform_image(sig, sr))
        spec_placeholder.image(mel_to_image(mel))

        model, classes = load_model()
        x = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0)
        probs = F.softmax(model(x), dim=1).detach().numpy()[0]

        idx = np.argmax(probs)
        disease = classes[idx]
        conf = probs[idx] * 100

        result_placeholder.markdown(
            f"### 🩺 Diagnosis: **{disease}**  \n"
            f"**Confidence:** `{conf:.2f}%`"
        )

        pdf = make_pdf(audio_name, disease, conf, mel, name, age, notes)
        b64 = base64.b64encode(pdf.read()).decode()

        st.markdown(
            f"<a href='data:application/pdf;base64,{b64}' download='lung_report.pdf'>📥 Download PDF Report</a>",
            unsafe_allow_html=True
        )
