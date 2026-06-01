
# app.py — Lung Sound Analyzer (Sky-Blue Theme)

import os
import io
import time
import base64
import tempfile
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
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

MODEL_PATH = "best_model.pth"
CLASSES_PATH = "classes.npy"
HISTORY_FILE = "history.csv"

SAMPLE_RATE = 22050
DURATION = 5.0
N_MELS = 128

st.set_page_config(page_title="Lung Sound — Sky Blue", layout="wide", initial_sidebar_state="expanded")

# ==========================
# SKY-BLUE THEME
# ==========================
def inject_css():
    st.markdown("""
    <style>

    html, body, [class*="css"], label, div, span, p, h1, h2, h3, h4, h5, h6 {
        color: #0b2239 !important;
        font-family: 'Inter', sans-serif;
    }

    /* Soft sky-blue medical gradient */
    .stApp {
        background: linear-gradient(
            180deg,
            #e3f5ff 0%,
            #cbecff 40%,
            #b6e2ff 70%,
            #a8dcff 100%
        );
        min-height: 100vh;
    }

    /* Header */
    .header {
        padding: 20px 26px;
        border-radius: 16px;
        background: rgba(255,255,255,0.4);
        border: 1px solid rgba(0,0,0,0.07);
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 20px;
        box-shadow: 0px 6px 22px rgba(0,0,0,0.15);
    }
    .logo {
        width: 58px; height: 58px;
        border-radius: 14px;
        background: linear-gradient(135deg,#0284c7,#0ea5e9);
        display:flex; align-items:center; justify-content:center;
        font-size:22px; font-weight:800; color:white;
        box-shadow: 0 4px 18px rgba(14,165,233,0.4);
    }
    .app-title { font-size: 22px; font-weight: 800; }
    .app-sub { color: #113854; opacity: 0.85; font-size: 14px; }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.55);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(0,0,0,0.07);
        backdrop-filter: blur(8px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        margin-bottom: 16px;
    }
    .card-strong { border-left: 6px solid #0ea5e9; }
    .section-title { font-size: 16px; font-weight: 700; margin-bottom: 12px; }
    .muted { font-size: 13px; color: #295b7a !important; }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,#0ea5e9,#0284c7);
        color: #fff;
        padding: 9px 15px;
        border-radius: 12px;
        border: none;
        font-weight: 700;
        box-shadow: 0 6px 20px rgba(14,165,233,0.35);
    }

    /* Upload box */
    .upload-box {
        border: 1px dashed rgba(0,0,0,0.2);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        background: rgba(255,255,255,0.5);
    }

    /* Wave/spectrogram containers */
    .wave, .spec {
        background: rgba(255,255,255,0.55);
        border-radius: 12px;
        padding: 8px;
    }

    /* Dataframe */
    .stDataFrame table { color: #0b2239 !important; }

    </style>
    """, unsafe_allow_html=True)

inject_css()

# -------------------------
# Header
# -------------------------
st.markdown("""
<div class="header">
    <div class="logo">LS</div>
    <div>
        <div class="app-title">Lung Sound Analysis</div>
        <div class="app-sub">Sky-Blue Medical Theme</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# MODEL
# -------------------------
class CNNClassifier(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
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
    m = CNNClassifier(len(classes))
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    return m, classes

def extract_features(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    y = y[:int(sr*DURATION)]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    log = librosa.power_to_db(mel, ref=np.max)
    log = (log-log.mean())/(log.std()+1e-9)
    return log, y, sr

def mel_to_image(log):
    arr = log - log.min()
    arr = arr / (arr.max() + 1e-9)
    return (arr * 255).astype(np.uint8)

def waveform_image(sig,sr):
    fig,ax = plt.subplots(figsize=(6,1.5))
    ax.plot(np.linspace(0,len(sig)/sr,len(sig)),sig,linewidth=0.7,color="#0284c7")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf,format="png",dpi=120,bbox_inches="tight",pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf

def make_pdf(file,disease,conf,mel,name,age,notes):
    buf = io.BytesIO()
    c = canvas.Canvas(buf,pagesize=A4)
    w,h = A4
    c.setFont("Helvetica-Bold",18)
    c.drawString(40,h-60,"Lung Sound Report")
    c.setFont("Helvetica",11)
    c.drawString(40,h-90,f"Patient: {name}")
    c.drawString(40,h-110,f"Age: {age}")
    c.drawString(40,h-130,f"Notes: {notes}")
    c.drawString(40,h-150,f"Disease: {disease}")
    c.drawString(40,h-170,f"Confidence: {conf:.2f}%")

    img_arr = mel_to_image(mel)
    pil = Image.fromarray(img_arr)
    imbuf = io.BytesIO()
    pil.save(imbuf,format="PNG")
    imbuf.seek(0)
    c.drawImage(ImageReader(imbuf),40,h-410, width=400, height=220)

    qr = qrcode.make(f"{name}|{disease}|{conf:.2f}")
    qbuf = io.BytesIO()
    qr.save(qbuf,format="PNG")
    qbuf.seek(0)
    c.drawImage(ImageReader(qbuf),40,h-650, width=110, height=110)

    c.save()
    buf.seek(0)
    return buf

# -------------------------
# UI
# -------------------------
left,right = st.columns([1.6,1])

with left:
    st.markdown('<div class="card card-strong">',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Audio</div>',unsafe_allow_html=True)
    uploaded = st.file_uploader(" ", type=["wav","mp3"])
    st.markdown("</div>",unsafe_allow_html=True)

    st.markdown('<div class="card">',unsafe_allow_html=True)
    name = st.text_input("Patient name")
    age = st.text_input("Age")
    notes = st.text_area("Notes")
    analyze = st.button("Analyze")
    st.markdown("</div>",unsafe_allow_html=True)

with right:
    res_wf = st.empty()
    res_sp = st.empty()
    res_txt = st.empty()

# -------------------------
# ANALYSIS
# -------------------------
if analyze:
    if not uploaded:
        st.error("Upload an audio file first.")
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".wav")
        tmp.write(uploaded.read())
        tmp.flush()

        mel, sig, sr = extract_features(tmp.name)
        res_wf.image(waveform_image(sig,sr))

        res_sp.image(mel_to_image(mel))

        model, classes = load_model()
        x = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0)
        out = model(x)
        probs = F.softmax(out,dim=1).detach().numpy()[0]
        idx = np.argmax(probs)
        disease = classes[idx]
        conf = probs[idx]*100

        res_txt.markdown(f"### Disease: **{disease}** — Confidence: **{conf:.2f}%**")

        pdf = make_pdf(uploaded.name,disease,conf,mel,name,age,notes)
        b64 = base64.b64encode(pdf.read()).decode()
        st.markdown(f"<a href='data:application/pdf;base64,{b64}' download='report.pdf'>📥 Download Report</a>",unsafe_allow_html=True)
