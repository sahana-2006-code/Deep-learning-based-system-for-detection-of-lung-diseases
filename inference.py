"""
=============================================================
  Multi-Label X-Ray Disease Detection — Inference v2.0
=============================================================
  Usage:
      python inference.py --image path/to/xray.jpg

  Options:
      --image      Path to X-ray image
      --model      Path to full checkpoint (default: best_model_full.pth)
      --show       Save and show prediction visualization
=============================================================
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models


# ─────────────────────────────────────────────
#  DEFAULTS (auto-loaded from checkpoint)
# ─────────────────────────────────────────────
DEFAULT_CHECKPOINT = "best_model_full.pth"

CLASS_COLORS = {
    "Tuberculosis" : "#E05C5C",
    "Pneumothorax" : "#5C9BE0",
    "Pneumonia"    : "#5CBE7A",
    "Normal"       : "#A0A0A0",
}


# ─────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────
def load_full_checkpoint(checkpoint_path):
    """Load model, thresholds, classes and image size from checkpoint."""
    ckpt        = torch.load(checkpoint_path, map_location="cpu")
    classes     = ckpt["classes"]
    thresholds  = ckpt["thresholds"]
    image_size  = ckpt["image_size"]
    num_classes = len(classes)

    # Rebuild EfficientNet-B2 with same classifier head
    model       = models.efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"✅ Checkpoint loaded: {checkpoint_path}")
    print(f"   Classes    : {classes}")
    print(f"   Thresholds : {[f'{t:.2f}' for t in thresholds]}")
    print(f"   Image size : {image_size}×{image_size}\n")

    return model, classes, thresholds, image_size


# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────
@torch.no_grad()
def predict(model, image_path, classes, thresholds, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    image  = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    logits = model(tensor)
    probs  = torch.sigmoid(logits).squeeze().numpy()

    prob_dict   = {cls: float(p) for cls, p in zip(classes, probs)}
    predictions = [cls for cls, p, t in zip(classes, probs, thresholds) if p >= t]

    return prob_dict, predictions, thresholds


# ─────────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────────
def visualize(image_path, prob_dict, predictions, thresholds, classes, save_path="prediction_result.png"):
    image = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#1C1C1E")

    # ── Left: X-ray ──
    axes[0].imshow(image, cmap="gray")
    axes[0].set_facecolor("#1C1C1E")
    axes[0].axis("off")
    title_text  = "Detected: " + ", ".join(predictions) if predictions else "No Disease Detected"
    title_color = "#E05C5C" if predictions and "Normal" not in predictions else "#5CBE7A"
    axes[0].set_title(title_text, color=title_color, fontsize=11, fontweight="bold", pad=10)

    # ── Right: probability bars ──
    axes[1].set_facecolor("#1C1C1E")
    probs_vals = list(prob_dict.values())
    colors     = [CLASS_COLORS.get(c, "#888888") for c in classes]
    alphas     = [1.0 if p >= t else 0.35 for p, t in zip(probs_vals, thresholds)]

    bars = axes[1].barh(classes, probs_vals, color=colors, height=0.5)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    # Per-class threshold lines
    for i, (cls, t) in enumerate(zip(classes, thresholds)):
        axes[1].plot([t, t], [i - 0.3, i + 0.3], color="white", linewidth=1.5, alpha=0.8)

    # Labels
    for i, (cls, prob) in enumerate(prob_dict.items()):
        axes[1].text(prob + 0.01, i, f"{prob:.1%}", va="center",
                     color="white", fontsize=10, fontweight="bold")

    axes[1].set_xlim(0, 1.18)
    axes[1].set_xlabel("Probability", color="white", fontsize=11)
    axes[1].set_title("Disease Probabilities\n(white lines = per-class thresholds)",
                       color="white", fontsize=11, pad=10)
    axes[1].tick_params(colors="white")
    axes[1].spines[["top", "right", "bottom", "left"]].set_visible(False)

    plt.suptitle("X-Ray Multi-Disease Analysis  —  EfficientNet-B2",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1C1C1E")
    print(f"  Saved visualization → {save_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="X-Ray Multi-Label Inference v2.0")
    parser.add_argument("--image", type=str, required=True,               help="Path to X-ray image")
    parser.add_argument("--model", type=str, default=DEFAULT_CHECKPOINT,  help="Path to full checkpoint")
    parser.add_argument("--show",  action="store_true",                   help="Save visualization")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"❌ Image not found: {args.image}"); return
    if not os.path.isfile(args.model):
        print(f"❌ Checkpoint not found: {args.model}")
        print("   Run train.py first!"); return

    model, classes, thresholds, image_size = load_full_checkpoint(args.model)
    prob_dict, predictions, thresholds     = predict(model, args.image, classes, thresholds, image_size)

    print("=" * 50)
    print("  PREDICTION RESULTS")
    print("=" * 50)
    print(f"  Image : {os.path.basename(args.image)}\n")
    for cls, prob, t in zip(classes, prob_dict.values(), thresholds):
        flag   = "  ← DETECTED" if prob >= t else ""
        marker = "🔴" if prob >= t else "⚪"
        print(f"  {marker} {cls:20s}: {prob:.2%}  (threshold: {t:.2f}){flag}")

    print("\n" + "─" * 50)
    if predictions:
        detected = [p for p in predictions if p != "Normal"]
        if detected:
            print(f"  ⚠️  Disease(s) Detected: {', '.join(detected)}")
        else:
            print("  ✅ Patient appears Normal")
    else:
        print("  ✅ No disease detected above threshold")
    print("=" * 50 + "\n")

    if args.show:
        visualize(args.image, prob_dict, predictions, thresholds, classes)


# ─────────────────────────────────────────────
#  BATCH INFERENCE
# ─────────────────────────────────────────────
def predict_batch(image_paths, checkpoint=DEFAULT_CHECKPOINT):
    model, classes, thresholds, image_size = load_full_checkpoint(checkpoint)
    results = []
    for path in image_paths:
        prob_dict, predictions, _ = predict(model, path, classes, thresholds, image_size)
        results.append((path, prob_dict, predictions))
    return results


if __name__ == "__main__":
    main()
