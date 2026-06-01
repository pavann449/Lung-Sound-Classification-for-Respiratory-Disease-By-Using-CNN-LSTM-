import os
import shutil
import pandas as pd
from tqdm import tqdm

# ==========================================================
# 🛠️ CONFIGURATION — CHANGE THESE PATHS AS NEEDED
# ==========================================================
CSV_PATH = r"C:\Users\rajsr\Desktop\proj\patient_diagnosis.csv"
AUDIO_DIR = r"C:\Users\rajsr\Desktop\proj\audio_and_txt_files"
OUTPUT_DIR = r"C:\Users\rajsr\Desktop\proj\dataset_by_disease"

# ==========================================================
# 🧾 STEP 1: Load the CSV (your file has NO headers)
# ==========================================================
df = pd.read_csv(CSV_PATH, header=None, names=["Patient_ID", "Diagnosis"])
print("✅ Loaded CSV with", len(df), "rows")
print("📄 First few rows:")
print(df.head())

# Create dictionary {patient_id: diagnosis}
patient_to_disease = dict(zip(df["Patient_ID"], df["Diagnosis"]))

# ==========================================================
# 📁 STEP 2: Create disease folders
# ==========================================================
diseases = sorted(df["Diagnosis"].unique())

for disease in diseases:
    folder_path = os.path.join(OUTPUT_DIR, disease)
    os.makedirs(folder_path, exist_ok=True)

print("\n✅ Created folders for diseases:")
print(", ".join(diseases))

# ==========================================================
# 🎵 STEP 3: Move or copy audio files
# ==========================================================
file_count = 0
missing_patients = []

for filename in tqdm(os.listdir(AUDIO_DIR), desc="Organizing"):
    if not filename.endswith(".wav"):
        continue

    # Extract patient ID (before the first underscore)
    try:
        patient_id = int(filename.split("_")[0])
    except ValueError:
        print(f"⚠️ Skipping file (invalid name): {filename}")
        continue

    disease = patient_to_disease.get(patient_id, None)
    if disease is None:
        missing_patients.append(patient_id)
        continue

    src_path = os.path.join(AUDIO_DIR, filename)
    dst_path = os.path.join(OUTPUT_DIR, disease, filename)

    # ✅ Use shutil.copy to keep original files intact
    # If you want to MOVE files instead, replace with: shutil.move(src_path, dst_path)
    shutil.copy(src_path, dst_path)
    file_count += 1

print(f"\n✅ Successfully organized {file_count} audio files!")

if missing_patients:
    print(f"⚠️ {len(missing_patients)} patient IDs not found in CSV: {sorted(set(missing_patients))}")

# ==========================================================
# 📊 STEP 4: Summary
# ==========================================================
print("\n📊 File count by disease:")
for disease in diseases:
    disease_dir = os.path.join(OUTPUT_DIR, disease)
    count = len([f for f in os.listdir(disease_dir) if f.endswith(".wav")])
    print(f"{disease:20s}: {count} files")
