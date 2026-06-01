import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ==========================================================
# ⚙️ CONFIGURATION
# ==========================================================
CSV_PATH = r"C:\Users\rajsr\Desktop\proj\features_index.csv"   # path to your preprocessed features
MODEL_PATH = r"C:\Users\rajsr\Desktop\proj\best_model.pth"
CLASSES_PATH = r"C:\Users\rajsr\Desktop\proj\classes.npy"

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4

# ==========================================================
# 🧩 Dataset Class
# ==========================================================
class LungSoundDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["path"])
        x = torch.FloatTensor(x).unsqueeze(0)  # shape: (1, n_mels, time)
        y = torch.tensor(row["label_idx"], dtype=torch.long)
        return x, y


# ==========================================================
# 🧠 Model Definition
# ==========================================================
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # ⚠️ Adjust this number if your input size changes
        self.fc1 = nn.Linear(64 * 16 * 27, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ==========================================================
# 📂 Load Dataset
# ==========================================================
df = pd.read_csv(CSV_PATH)
print("✅ Loaded features CSV with", len(df), "samples")

# Show class distribution
print("\n📊 Original class distribution:")
print(df["label"].value_counts())

# ==========================================================
# 🩺 Handle Rare Classes
# ==========================================================
min_count = df["label"].value_counts().min()
if min_count < 2:
    print("\n⚠️ Some classes have fewer than 2 samples — removing them for stable training...")
    df = df.groupby("label").filter(lambda x: len(x) >= 2)

print("\n✅ Cleaned class distribution:")
print(df["label"].value_counts())

# Encode labels
le = LabelEncoder()
df["label_idx"] = le.fit_transform(df["label"])
np.save(CLASSES_PATH, le.classes_)
num_classes = len(le.classes_)
print("\n🧾 Classes:", list(le.classes_))

# ==========================================================
# ✂️ Train/Validation Split
# ==========================================================
if df["label"].value_counts().min() < 2:
    print("⚠️ Using non-stratified split (some classes are very small).")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
else:
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label_idx"], random_state=42)

print(f"\n✅ Training samples: {len(train_df)} | Validation samples: {len(val_df)}")

# ==========================================================
# 📦 Create Dataloaders
# ==========================================================
train_ds = LungSoundDataset(train_df)
val_ds = LungSoundDataset(val_df)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ==========================================================
# 🚀 Training Setup
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️ Using device: {device}")

model = CNNClassifier(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ==========================================================
# 🎯 Training Loop
# ==========================================================
best_acc = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # ---- Validation ----
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch}: Train Loss={total_loss/len(train_dl):.4f}, Val Acc={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"💾 Saved new best model with accuracy {best_acc:.4f}")

print("\n🎉 Training complete!")
print(f"✅ Best validation accuracy: {best_acc:.4f}")
print(f"📁 Model saved to: {MODEL_PATH}")
print(f"🧾 Classes saved to: {CLASSES_PATH}")
