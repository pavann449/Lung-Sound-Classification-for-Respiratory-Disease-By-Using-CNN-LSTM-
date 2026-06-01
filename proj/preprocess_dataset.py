import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
DATASET_DIR = r"C:\Users\rajsr\Desktop\proj\dataset_by_disease"
OUTPUT_FEATURE_DIR = r"C:\Users\rajsr\Desktop\proj\processed_features"
CSV_OUTPUT = r"C:\Users\rajsr\Desktop\proj\features_index.csv"

SAMPLE_RATE = 22050
DURATION = 5.0  # seconds
N_MELS = 128

os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)

data_records = []

def extract_mel(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.astype(np.float32)

# === Process each disease folder ===
for disease in os.listdir(DATASET_DIR):
    disease_dir = os.path.join(DATASET_DIR, disease)
    if not os.path.isdir(disease_dir):
        continue

    print(f"🎧 Processing {disease} ...")

    for file in tqdm(os.listdir(disease_dir)):
        if not file.endswith(".wav"):
            continue
        file_path = os.path.join(disease_dir, file)
        mel = extract_mel(file_path)

        # Save mel features
        save_name = file.replace(".wav", ".npy")
        save_path = os.path.join(OUTPUT_FEATURE_DIR, save_name)
        np.save(save_path, mel)

        data_records.append({
            "file": save_name,
            "label": disease,
            "path": save_path
        })

# === Save CSV index ===
df = pd.DataFrame(data_records)
df.to_csv(CSV_OUTPUT, index=False)
print(f"\n✅ Done! Extracted {len(df)} files.")
print(f"📁 Saved features in: {OUTPUT_FEATURE_DIR}")
print(f"🧾 Index file: {CSV_OUTPUT}")
