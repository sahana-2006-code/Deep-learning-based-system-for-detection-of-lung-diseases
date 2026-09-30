"""
================================================================
  Generate Evaluation Metrics from Saved Model
  - Per-class thresholds + Val F1 + Val AUC-ROC table
  - ROC curves per class
  - Training curves (if history file exists)
================================================================
  Run: python generate_metrics.py
  Place in same folder as best_model_full.pth and dataset/
================================================================
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    f1_score, roc_auc_score, roc_curve, auc
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ──────────────────────────────────────────────
#  CONFIG — adjust if your folder names differ
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_full.pth")
DATA_DIR   = os.path.join(BASE_DIR, "dataset")
OUT_DIR    = BASE_DIR   # saves images here

VAL_SPLIT  = 0.15       # must match what you used in train.py
SEED       = 42
BATCH_SIZE = 16

# ──────────────────────────────────────────────
#  LOAD MODEL
# ──────────────────────────────────────────────
print("\n📦 Loading model checkpoint...")
ckpt       = torch.load(MODEL_PATH, map_location="cpu",weights_only=False)
CLASSES    = ckpt["classes"]
THRESHOLDS = ckpt["thresholds"]
IMAGE_SIZE = ckpt["image_size"]
N          = len(CLASSES)
print(f"   Classes    : {CLASSES}")
print(f"   Thresholds : {[round(t,2) for t in THRESHOLDS]}")
print(f"   Image size : {IMAGE_SIZE}")

def build_model(n):
    m = models.efficientnet_b2(weights=None)
    inf = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.BatchNorm1d(inf), nn.Dropout(0.5),
        nn.Linear(inf, 256), nn.SiLU(),
        nn.BatchNorm1d(256), nn.Dropout(0.3),
        nn.Linear(256, n),
    )
    return m

model = build_model(N)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("   Model loaded OK\n")

# ──────────────────────────────────────────────
#  DATASET
# ──────────────────────────────────────────────
class XRayDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

def build_samples(data_dir, classes):
    valid = {".jpg",".jpeg",".png",".bmp",".tiff"}
    samples = []
    for ci, cn in enumerate(classes):
        folder = os.path.join(data_dir, cn)
        if not os.path.isdir(folder):
            print(f"   WARNING: folder not found: {folder}")
            continue
        lv = [0.0]*len(classes)
        lv[ci] = 1.0
        for f in sorted(os.listdir(folder)):
            if os.path.splitext(f)[1].lower() in valid:
                samples.append((os.path.join(folder, f), lv[:]))
    return samples

def split_dataset(samples, val_ratio=0.15, seed=42):
    from collections import defaultdict
    buckets = defaultdict(list)
    for s in samples:
        buckets[int(np.argmax(s[1]))].append(s)
    val_s = []
    rng = np.random.default_rng(seed)
    for bucket in buckets.values():
        bucket = list(bucket)
        rng.shuffle(bucket)
        n    = len(bucket)
        va_n = int(n * val_ratio)
        val_s += bucket[:va_n]
    rng.shuffle(val_s)
    return val_s

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

print("📂 Loading dataset...")
all_samples = build_samples(DATA_DIR, CLASSES)
print(f"   Total images : {len(all_samples)}")
val_samples = split_dataset(all_samples, VAL_SPLIT, SEED)
print(f"   Val samples  : {len(val_samples)}\n")

val_ds     = XRayDataset(val_samples, transform)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ──────────────────────────────────────────────
#  COLLECT PREDICTIONS
# ──────────────────────────────────────────────
print("🔄 Running inference on validation set...")
all_probs  = []
all_labels = []

with torch.no_grad():
    for imgs, labels in val_loader:
        probs = torch.sigmoid(model(imgs)).numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

all_probs  = np.vstack(all_probs)   # shape: [N_val, N_classes]
all_labels = np.vstack(all_labels)  # shape: [N_val, N_classes]
print(f"   Done — {len(all_probs)} samples evaluated\n")

# ──────────────────────────────────────────────
#  1. THRESHOLD TABLE
# ──────────────────────────────────────────────
print("=" * 60)
print("  PER-CLASS THRESHOLD EVALUATION TABLE")
print("=" * 60)
print(f"  {'Class':<18} {'Threshold':>10} {'Val F1':>10} {'Val AUC-ROC':>12}")
print("  " + "-"*52)

table_data = []
for i, cls in enumerate(CLASSES):
    t   = THRESHOLDS[i]
    # Val F1 at tuned threshold
    preds_t   = (all_probs[:, i] >= t).astype(int)
    f1_t      = f1_score(all_labels[:, i], preds_t, zero_division=0)
    # Val AUC-ROC
    auc_val   = roc_auc_score(all_labels[:, i], all_probs[:, i])
    table_data.append((cls, t, f1_t, auc_val))
    print(f"  {cls:<18} {t:>10.2f} {f1_t:>10.4f} {auc_val:>12.4f}")

print("=" * 60)

# Also show what F1 would be at default 0.50
print("\n  COMPARISON: Tuned vs Default threshold (0.50)")
print("=" * 60)
print(f"  {'Class':<18} {'Default F1':>12} {'Tuned F1':>10} {'Improvement':>13}")
print("  " + "-"*55)
for cls, t, f1_t, auc_val in table_data:
    i = CLASSES.index(cls)
    preds_default = (all_probs[:, i] >= 0.50).astype(int)
    f1_default    = f1_score(all_labels[:, i], preds_default, zero_division=0)
    improvement   = f1_t - f1_default
    sign          = "+" if improvement >= 0 else ""
    print(f"  {cls:<18} {f1_default:>12.4f} {f1_t:>10.4f} {sign}{improvement:>12.4f}")
print("=" * 60)

# ──────────────────────────────────────────────
#  2. ROC CURVES — one plot with all 4 classes
# ──────────────────────────────────────────────
print("\n📈 Generating ROC curves...")

COLORS = {
    "Tuberculosis": "#E53935",
    "Pneumonia":    "#1E88E5",
    "Pneumothorax": "#00897B",
    "Normal":       "#8E24AA",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("white")

# ── Left plot: All classes on one graph ──
ax1 = axes[0]
ax1.set_facecolor("#F8FBFF")
ax1.plot([0,1],[0,1], color="#CFD8DC", linestyle="--",
         linewidth=1.2, label="Random Classifier")

for cls in CLASSES:
    i         = CLASSES.index(cls)
    fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
    roc_auc   = auc(fpr, tpr)
    color     = COLORS.get(cls, "#607D8B")
    ax1.plot(fpr, tpr, color=color, linewidth=2.5,
             label=f"{cls}  (AUC = {roc_auc:.4f})")

ax1.set_xlabel("False Positive Rate", fontsize=12, color="#37474F")
ax1.set_ylabel("True Positive Rate", fontsize=12, color="#37474F")
ax1.set_title("ROC Curves — All Classes", fontsize=13,
              fontweight="bold", color="#1E3A5F", pad=12)
ax1.legend(fontsize=10, framealpha=0.9, loc="lower right")
ax1.set_xlim([-0.01, 1.01]); ax1.set_ylim([-0.01, 1.01])
ax1.spines[["top","right"]].set_visible(False)
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.tick_params(colors="#607D8B")

# ── Right plot: Individual subplots per class ──
ax2 = axes[1]
ax2.set_visible(False)

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
fig2.patch.set_facecolor("white")
fig2.suptitle("Per-Class ROC Curves", fontsize=15,
              fontweight="bold", color="#1E3A5F", y=1.01)

for idx, cls in enumerate(CLASSES):
    row, col = idx // 2, idx % 2
    ax  = axes2[row][col]
    ax.set_facecolor("#F8FBFF")
    i   = CLASSES.index(cls)
    fpr, tpr, thresholds_roc = roc_curve(all_labels[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    color   = COLORS.get(cls, "#607D8B")

    # Shade area under curve
    ax.fill_between(fpr, tpr, alpha=0.12, color=color)
    ax.plot(fpr, tpr, color=color, linewidth=2.5,
            label=f"AUC = {roc_auc:.4f}")
    ax.plot([0,1],[0,1], color="#CFD8DC", linestyle="--", linewidth=1)

    # Mark the operating point at tuned threshold
    t     = THRESHOLDS[i]
    diffs = np.abs(thresholds_roc - t)
    op_idx = np.argmin(diffs)
    ax.scatter(fpr[op_idx], tpr[op_idx], color=color,
               s=80, zorder=5, label=f"Threshold = {t:.2f}")

    ax.set_xlabel("False Positive Rate", fontsize=10, color="#546E7A")
    ax.set_ylabel("True Positive Rate",  fontsize=10, color="#546E7A")
    ax.set_title(cls, fontsize=12, fontweight="bold", color=color, pad=8)
    ax.legend(fontsize=9, framealpha=0.9, loc="lower right")
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(colors="#607D8B")

plt.tight_layout()

# Save
roc_all_path = os.path.join(OUT_DIR, "roc_curves_all.png")
roc_per_path = os.path.join(OUT_DIR, "roc_curves_perclass.png")

fig.axes[0].figure.savefig(roc_all_path,  dpi=180, bbox_inches="tight",
                            facecolor="white")
fig2.savefig(roc_per_path, dpi=180, bbox_inches="tight", facecolor="white")

plt.close("all")
print(f"   Saved → roc_curves_all.png")
print(f"   Saved → roc_curves_perclass.png")

# ──────────────────────────────────────────────
#  3. TRAINING CURVES — from saved history JSON
# ──────────────────────────────────────────────
history_path = os.path.join(BASE_DIR, "training_history.json")
curves_png   = os.path.join(OUT_DIR,  "training_curves.png")

if os.path.exists(history_path):
    print("\n📉 Generating training curves from saved history...")
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.patch.set_facecolor("white")

    for ax in axes3:
        ax.set_facecolor("#F8FBFF")
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(colors="#607D8B")

    # Loss
    axes3[0].plot(epochs, history["train_loss"], label="Train Loss",
                  linewidth=2.5, color="#1E88E5")
    axes3[0].plot(epochs, history["val_loss"],   label="Val Loss",
                  linewidth=2.5, color="#E53935")
    axes3[0].fill_between(epochs, history["train_loss"], history["val_loss"],
                           alpha=0.06, color="#9C27B0")
    axes3[0].set_title("Training vs Validation Loss",
                        fontsize=13, fontweight="bold", color="#1E3A5F", pad=12)
    axes3[0].set_xlabel("Epoch", fontsize=11, color="#546E7A")
    axes3[0].set_ylabel("BCEWithLogits Loss", fontsize=11, color="#546E7A")
    axes3[0].legend(fontsize=11)

    # F1
    axes3[1].plot(epochs, history["train_f1"], label="Train F1",
                  linewidth=2.5, color="#1E88E5")
    axes3[1].plot(epochs, history["val_f1"],   label="Val F1",
                  linewidth=2.5, color="#E53935")
    axes3[1].fill_between(epochs, history["train_f1"], history["val_f1"],
                           alpha=0.06, color="#9C27B0")
    axes3[1].set_title("Training vs Validation F1 Score",
                        fontsize=13, fontweight="bold", color="#1E3A5F", pad=12)
    axes3[1].set_xlabel("Epoch", fontsize=11, color="#546E7A")
    axes3[1].set_ylabel("Macro F1 Score",      fontsize=11, color="#546E7A")
    axes3[1].set_ylim(0, 1)
    axes3[1].legend(fontsize=11)

    plt.suptitle("EfficientNet-B2 | Training History",
                 fontsize=14, fontweight="bold", color="#1E3A5F", y=1.02)
    plt.tight_layout()
    fig3.savefig(curves_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   Saved → training_curves.png")
else:
    print(f"\n⚠️  training_history.json not found.")
    print("   To generate training curves, add this to the END of your train.py")
    print("   main() function — just before the final print statement:\n")
    print("   import json")
    print("   with open('training_history.json', 'w') as f:")
    print("       json.dump(history, f)")
    print("\n   Then retrain once and re-run this script.")

# ──────────────────────────────────────────────
#  DONE
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ALL DONE")
print("=" * 60)
print(f"  roc_curves_all.png      → ROC all classes on one plot")
print(f"  roc_curves_perclass.png → ROC one subplot per class")
if os.path.exists(history_path):
    print(f"  training_curves.png     → Loss + F1 training history")
print("\n  Use these images directly in your report / presentation.")
print("=" * 60 + "\n")
