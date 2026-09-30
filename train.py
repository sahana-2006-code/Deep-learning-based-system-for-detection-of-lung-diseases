"""
=============================================================
  Multi-Label X-Ray Disease Detection — OPTIMIZED v2.0
=============================================================
  Classes  : Tuberculosis | Pneumothorax | Pneumonia | Normal
  Dataset  : 2100 images per class (8400 total)
  Model    : EfficientNet-B2 (pretrained ImageNet)
  Strategy : Single-label training → Multi-label inference
  Loss     : BCEWithLogitsLoss + Label Smoothing
  Extras   : Mixup, Cosine LR, Differential LR,
             Per-class threshold tuning
=============================================================
"""

import os
import copy
import time
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    precision_score, recall_score, accuracy_score,
    confusion_matrix as sk_confusion_matrix
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    # ── Data ──
    "data_dir"            : "dataset",
    "classes"             : ["Tuberculosis", "Pneumothorax", "Pneumonia", "Normal"],
    "split"               : (0.70, 0.15, 0.15),

    # ── Model ──
    "image_size"          : 260,         # EfficientNet-B2 native resolution
    "model_name"          : "EfficientNet-B2",

    # ── Training ──
    "batch_size"          : 16,
    "num_epochs"          : 60,
    "warmup_epochs"       : 5,
    "learning_rate"       : 5e-5,
    "weight_decay"        : 1e-4,
    "early_stop_patience" : 10,
    "label_smoothing"     : 0.1,
    "grad_clip"           : 1.0,

    # ── Mixup ──
    "mixup_alpha"         : 0.3,
    "mixup_prob"          : 0.5,

    # ── Inference ──
    "threshold"           : 0.40,

    # ── Output ──
    "checkpoint_path"     : "best_model.pth",
    "full_save_path"      : "best_model_full.pth",
    "seed"                : 42,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
DEVICE = torch.device("cpu")

print("\n" + "═" * 65)
print("  MULTI-LABEL X-RAY DISEASE DETECTION  —  v2.0")
print("═" * 65)
print(f"  Device   : {DEVICE}")
print(f"  Model    : {CONFIG['model_name']}")
print(f"  Classes  : {CONFIG['classes']}")
print(f"  Image sz : {CONFIG['image_size']}×{CONFIG['image_size']}")
print("═" * 65 + "\n")


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────
class XRayDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_vec = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label_vec, dtype=torch.float32)


def build_samples(data_dir, classes):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    samples    = []

    print("📂 Loading dataset:")
    print("─" * 45)
    for class_idx, class_name in enumerate(classes):
        folder = os.path.join(data_dir, class_name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"\n❌ Folder not found: '{folder}'\n"
                f"   Name must match exactly: '{class_name}'"
            )
        label_vec             = [0.0] * len(classes)
        label_vec[class_idx]  = 1.0
        count = 0
        for fname in sorted(os.listdir(folder)):
            if os.path.splitext(fname)[1].lower() in valid_exts:
                samples.append((os.path.join(folder, fname), label_vec[:]))
                count += 1
        print(f"   ✅ {class_name:20s}: {count:>5} images")

    print("─" * 45)
    print(f"   Total                  : {len(samples):>5} images\n")
    return samples


def get_transforms(image_size, mode="train"):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def split_dataset(samples, split_ratios, seed=42):
    """True stratified split — balanced per class."""
    class_buckets = defaultdict(list)
    for s in samples:
        class_buckets[int(np.argmax(s[1]))].append(s)

    train_s, val_s, test_s = [], [], []
    rng = np.random.default_rng(seed)

    for bucket in class_buckets.values():
        bucket = list(bucket)
        rng.shuffle(bucket)
        n        = len(bucket)
        tr_n     = int(n * split_ratios[0])
        va_n     = int(n * split_ratios[1])
        train_s += bucket[:tr_n]
        val_s   += bucket[tr_n:tr_n + va_n]
        test_s  += bucket[tr_n + va_n:]

    rng.shuffle(train_s)
    rng.shuffle(val_s)
    rng.shuffle(test_s)

    print(f"📊 Stratified Split:")
    print(f"   Train : {len(train_s)} | Val : {len(val_s)} | Test : {len(test_s)}\n")
    return train_s, val_s, test_s


# ─────────────────────────────────────────────
#  MIXUP AUGMENTATION
# ─────────────────────────────────────────────
def mixup_batch(images, labels, alpha=0.3):
    lam   = np.random.beta(alpha, alpha)
    idx   = torch.randperm(images.size(0))
    return (
        lam * images + (1 - lam) * images[idx],
        lam * labels + (1 - lam) * labels[idx]
    )


# ─────────────────────────────────────────────
#  MODEL — EfficientNet-B2
# ─────────────────────────────────────────────
def build_model(num_classes):
    model       = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


def unfreeze_model(model, lr, optimizer):
    for param in model.parameters():
        param.requires_grad = True

    backbone_params   = [p for n, p in model.named_parameters() if "classifier" not in n]
    classifier_params = [p for n, p in model.named_parameters() if "classifier" in n]

    optimizer.param_groups.clear()
    optimizer.add_param_group({"params": backbone_params,   "lr": lr / 10})
    optimizer.add_param_group({"params": classifier_params, "lr": lr})

    print("  🔓 Backbone unfrozen — differential LR:")
    print(f"     Backbone LR   : {lr/10:.2e}")
    print(f"     Classifier LR : {lr:.2e}")


# ─────────────────────────────────────────────
#  LABEL SMOOTHING LOSS
# ─────────────────────────────────────────────
class SmoothBCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce       = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        smooth_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, smooth_targets)


# ─────────────────────────────────────────────
#  PROGRESS BAR
# ─────────────────────────────────────────────
def print_progress(batch_idx, total_batches, loss, f1, prefix="Train"):
    pct    = (batch_idx + 1) / total_batches
    bar    = "█" * int(28 * pct) + "░" * (28 - int(28 * pct))
    sys.stdout.write(
        f"\r  {prefix} [{bar}] {pct*100:5.1f}%  "
        f"Batch {batch_idx+1}/{total_batches}  "
        f"Loss: {loss:.4f}  F1: {f1:.4f}"
    )
    sys.stdout.flush()


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, epoch, total_epochs):
    model.train()
    running_loss          = 0.0
    all_preds, all_labels = [], []
    total_batches         = len(loader)

    print(f"\n  Epoch {epoch}/{total_epochs}")

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if CONFIG["mixup_alpha"] > 0 and np.random.rand() < CONFIG["mixup_prob"]:
            images, labels = mixup_batch(images, labels, CONFIG["mixup_alpha"])

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        with torch.no_grad():
            probs       = torch.sigmoid(outputs).cpu().numpy()
            hard_labels = (labels.cpu().numpy() >= 0.5).astype(int)
            all_preds.append((probs >= CONFIG["threshold"]).astype(int))
            all_labels.append(hard_labels)

        batch_f1 = f1_score(
            np.vstack(all_labels), np.vstack(all_preds),
            average="macro", zero_division=0
        )
        print_progress(batch_idx, total_batches,
                       running_loss / ((batch_idx + 1) * CONFIG["batch_size"]),
                       batch_f1, "  🔵 Train")

    print()
    epoch_loss = running_loss / len(loader.dataset)
    f1 = f1_score(np.vstack(all_labels), np.vstack(all_preds), average="macro", zero_division=0)
    return epoch_loss, f1


@torch.no_grad()
def evaluate(model, loader, criterion, show_progress=False):
    model.eval()
    running_loss                     = 0.0
    all_probs, all_preds, all_labels = [], [], []
    total_batches                    = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs        = model(images)
        running_loss  += criterion(outputs, labels).item() * images.size(0)

        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.append(probs)
        all_preds.append((probs >= CONFIG["threshold"]).astype(int))
        all_labels.append(labels.cpu().numpy().astype(int))

        if show_progress:
            print_progress(batch_idx, total_batches,
                           running_loss / ((batch_idx + 1) * CONFIG["batch_size"]),
                           0.0, "  🟡 Val  ")
    if show_progress:
        print()

    probs = np.vstack(all_probs)
    preds = np.vstack(all_preds)
    labs  = np.vstack(all_labels)
    f1    = f1_score(labs, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labs, probs, average="macro")
    except ValueError:
        auc = float("nan")

    return running_loss / len(loader.dataset), f1, auc, preds, labs, probs


# ─────────────────────────────────────────────
#  PER-CLASS THRESHOLD TUNING
# ─────────────────────────────────────────────
def tune_thresholds(model, val_loader, criterion, classes):
    print("\n🎯 Tuning per-class thresholds on validation set...")
    _, _, _, _, labs, probs = evaluate(model, val_loader, criterion)

    thresholds      = np.arange(0.25, 0.76, 0.05)
    best_thresholds = []

    for i, cls in enumerate(classes):
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            f1_t = f1_score(labs[:, i], (probs[:, i] >= t).astype(int), zero_division=0)
            if f1_t > best_f1:
                best_f1, best_t = f1_t, t
        best_thresholds.append(best_t)
        print(f"   {cls:20s}: threshold = {best_t:.2f}  (F1 = {best_f1:.4f})")

    return best_thresholds


@torch.no_grad()
def evaluate_with_thresholds(model, loader, thresholds):
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in loader:
        probs = torch.sigmoid(model(images.to(DEVICE))).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy().astype(int))

    probs = np.vstack(all_probs)
    labs  = np.vstack(all_labels)
    preds = np.stack(
        [(probs[:, i] >= t).astype(int) for i, t in enumerate(thresholds)], axis=1
    )
    return preds, labs, probs


# ─────────────────────────────────────────────
#  MAIN TRAIN FUNCTION
# ─────────────────────────────────────────────
def train(model, train_loader, val_loader, criterion, optimizer, scheduler):
    history           = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_val_f1       = 0.0
    best_weights      = None
    patience_cnt      = 0
    total_epochs      = CONFIG["num_epochs"]
    backbone_unfrozen = False

    print("\n" + "═" * 65)
    print("  TRAINING CONFIGURATION")
    print("─" * 65)
    print(f"  Epochs          : {total_epochs}  (warmup: {CONFIG['warmup_epochs']})")
    print(f"  Learning rate   : {CONFIG['learning_rate']:.2e}")
    print(f"  Weight decay    : {CONFIG['weight_decay']:.2e}")
    print(f"  Label smoothing : {CONFIG['label_smoothing']}")
    print(f"  Mixup alpha     : {CONFIG['mixup_alpha']}  (prob: {CONFIG['mixup_prob']})")
    print(f"  Early stopping  : {CONFIG['early_stop_patience']} epochs patience")
    print(f"  Threshold       : {CONFIG['threshold']}")
    print("═" * 65)

    for epoch in range(1, total_epochs + 1):

        # Overall progress bar
        pct = (epoch - 1) / total_epochs * 100
        bar = "█" * int(28 * (epoch-1) / total_epochs) + "░" * (28 - int(28 * (epoch-1) / total_epochs))
        print(f"\n  Overall  [{bar}]  {pct:.1f}%  (Epoch {epoch}/{total_epochs})")
        print("  " + "─" * 63)

        # Unfreeze after warmup
        if epoch == CONFIG["warmup_epochs"] + 1 and not backbone_unfrozen:
            unfreeze_model(model, CONFIG["learning_rate"], optimizer)
            backbone_unfrozen = True

        t_start      = time.time()
        t_loss, t_f1 = train_one_epoch(model, train_loader, criterion, optimizer, epoch, total_epochs)
        v_loss, v_f1, v_auc, _, _, _ = evaluate(model, val_loader, criterion, show_progress=True)
        scheduler.step()
        elapsed = time.time() - t_start

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_f1"].append(t_f1)
        history["val_f1"].append(v_f1)

        is_best   = v_f1 > best_val_f1
        medal     = "🏆 BEST" if is_best else "      "
        gap       = t_f1 - v_f1
        gap_warn  = " ⚠️ overfit" if gap > 0.12 else ""
        f1_delta  = v_f1 - (history["val_f1"][-2] if len(history["val_f1"]) > 1 else v_f1)
        delta_str = f"{'↑' if f1_delta >= 0 else '↓'}{abs(f1_delta):.4f}"

        print(f"\n  ┌{'─'*61}┐")
        print(f"  │  Epoch {epoch:>3}/{total_epochs}   {medal}   Time: {elapsed:.1f}s                   │")
        print(f"  ├{'─'*61}┤")
        print(f"  │  Train →  Loss: {t_loss:.4f}   F1: {t_f1:.4f} ({t_f1*100:.2f}%)              │")
        print(f"  │  Val   →  Loss: {v_loss:.4f}   F1: {v_f1:.4f} ({v_f1*100:.2f}%)  {delta_str:<10}  │")
        print(f"  │  AUC: {v_auc:.4f}   Gap: {gap:.4f}{gap_warn:<12}  Patience: {patience_cnt}/{CONFIG['early_stop_patience']}  │")
        print(f"  └{'─'*61}┘")

        if is_best:
            best_val_f1  = v_f1
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, CONFIG["checkpoint_path"])
            patience_cnt = 0
            print(f"  💾 Saved → {CONFIG['checkpoint_path']}")
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["early_stop_patience"]:
                print(f"\n  ⏹  Early stopping at epoch {epoch}")
                print(f"     Best Val F1 = {best_val_f1:.4f} ({best_val_f1*100:.2f}%)")
                break

    print(f"\n  Overall  [{'█'*28}]  100.0%  Complete!")
    print("\n" + "═" * 65)
    print(f"  ✅ Training Complete!  Best Val F1 = {best_val_f1:.4f} ({best_val_f1*100:.2f}%)")
    print("═" * 65)

    model.load_state_dict(best_weights)
    return model, history


# ─────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#F8F9FA")
    epochs = range(1, len(history["train_loss"]) + 1)

    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2.5, color="#2196F3")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   linewidth=2.5, color="#F44336")
    axes[0].fill_between(epochs, history["train_loss"], history["val_loss"], alpha=0.08, color="#9C27B0")
    axes[0].set_title("BCEWithLogits Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].legend(fontsize=11)

    axes[1].plot(epochs, history["train_f1"], label="Train", linewidth=2.5, color="#2196F3")
    axes[1].plot(epochs, history["val_f1"],   label="Val",   linewidth=2.5, color="#F44336")
    axes[1].fill_between(epochs, history["train_f1"], history["val_f1"], alpha=0.08, color="#9C27B0")
    axes[1].set_title("Macro F1 Score", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].legend(fontsize=11)
    axes[1].set_ylim(0, 1)

    plt.suptitle(f"{CONFIG['model_name']} | X-Ray Multi-Label Classification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    print("  Saved → training_curves.png")


# ─────────────────────────────────────────────
#  FULL EVALUATION
# ─────────────────────────────────────────────
def full_evaluation(model, test_loader, val_loader, criterion, classes):
    print("\n" + "═" * 65)
    print("  TEST SET EVALUATION")
    print("═" * 65)

    # Tune thresholds on val set
    best_thresholds = tune_thresholds(model, val_loader, criterion, classes)

    # Evaluate on test set with tuned thresholds
    preds, labs, probs = evaluate_with_thresholds(model, test_loader, best_thresholds)

    acc       = accuracy_score(labs, preds)
    precision = precision_score(labs, preds, average="macro", zero_division=0)
    recall    = recall_score(labs, preds,    average="macro", zero_division=0)
    f1        = f1_score(labs, preds,        average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labs, probs, average="macro")
    except ValueError:
        auc = float("nan")

    print(f"\n  {'Metric':<24} {'Score':>8}  {'%':>8}")
    print(f"  {'─'*44}")
    print(f"  {'Accuracy (exact match)':<24} {acc:>8.4f}  {acc*100:>7.2f}%")
    print(f"  {'Macro Precision':<24} {precision:>8.4f}  {precision*100:>7.2f}%")
    print(f"  {'Macro Recall':<24} {recall:>8.4f}  {recall*100:>7.2f}%")
    print(f"  {'Macro F1':<24} {f1:>8.4f}  {f1*100:>7.2f}%")
    print(f"  {'Macro AUC-ROC':<24} {auc:>8.4f}  {auc*100:>7.2f}%")

    # Per-class report
    print("\n" + "═" * 65)
    print("  PER-CLASS REPORT")
    print("═" * 65)
    print(classification_report(labs, preds, target_names=classes, zero_division=0))

    # Per-class AUC with visual bar
    print("═" * 65)
    print("  PER-CLASS AUC-ROC")
    print("═" * 65)
    for i, cls in enumerate(classes):
        try:
            cls_auc = roc_auc_score(labs[:, i], probs[:, i])
            bar     = "█" * int(cls_auc * 20) + "░" * (20 - int(cls_auc * 20))
            print(f"  {cls:20s}: {cls_auc:.4f}  [{bar}]  {cls_auc*100:.1f}%")
        except ValueError:
            print(f"  {cls:20s}: N/A")

    # Confusion matrix
    print("\n" + "═" * 65)
    print("  COMBINED CONFUSION MATRIX")
    print("═" * 65)

    true_single = np.argmax(labs,  axis=1)
    pred_single = np.argmax(probs, axis=1)
    cm          = sk_confusion_matrix(true_single, pred_single, labels=list(range(len(classes))))

    header = f"  {'':>18} " + "  ".join(f"{c[:12]:>12}" for c in classes)
    print(header)
    for i, row_cls in enumerate(classes):
        row_str = "  ".join(f"{val:>12}" for val in cm[i])
        print(f"  {row_cls[:18]:>18} {row_str}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#F8F9FA")
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label",      fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("\n  Saved → confusion_matrix.png")

    # Tuned threshold summary
    print("\n" + "═" * 65)
    print("  PER-CLASS TUNED THRESHOLDS (used for final predictions)")
    print("═" * 65)
    for cls, t in zip(classes, best_thresholds):
        bar = "█" * int(t * 20) + "░" * (20 - int(t * 20))
        print(f"  {cls:20s}: {t:.2f}  [{bar}]")

    print("\n" + "═" * 65)
    print("  ✅ All evaluation files saved!")
    print("═" * 65)

    return best_thresholds


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    # ── 1. Data ──
    samples = build_samples(CONFIG["data_dir"], CONFIG["classes"])
    train_s, val_s, test_s = split_dataset(samples, CONFIG["split"], CONFIG["seed"])

    sz       = CONFIG["image_size"]
    train_ds = XRayDataset(train_s, get_transforms(sz, "train"))
    val_ds   = XRayDataset(val_s,   get_transforms(sz, "val"))
    test_ds  = XRayDataset(test_s,  get_transforms(sz, "test"))

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    # ── 2. Model ──
    print(f"🏗️  Building {CONFIG['model_name']} ...")
    model     = build_model(num_classes=len(CONFIG["classes"]))
    criterion = SmoothBCELoss(smoothing=CONFIG["label_smoothing"])
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-7
    )

    # ── 3. Train ──
    model, history = train(model, train_loader, val_loader, criterion, optimizer, scheduler)

    # ── 4. Plot ──
    plot_history(history)

    # ── 5. Evaluate ──
    best_thresholds = full_evaluation(
        model, test_loader, val_loader, criterion, CONFIG["classes"]
    )

    # ── 6. Save full checkpoint ──
    torch.save({
        "model_state_dict" : model.state_dict(),
        "thresholds"       : best_thresholds,
        "classes"          : CONFIG["classes"],
        "image_size"       : CONFIG["image_size"],
        "model_name"       : CONFIG["model_name"],
    }, CONFIG["full_save_path"])

    print(f"\n✅ All done!")
    print(f"   Checkpoint  → {CONFIG['checkpoint_path']}")
    print(f"   Full save   → {CONFIG['full_save_path']}")
    print(f"   Curves      → training_curves.png")
    print(f"   Matrix      → confusion_matrix.png\n")


if __name__ == "__main__":
    main()
