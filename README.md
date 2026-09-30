# 🫁 Multi-Label X-Ray Disease Detection — v2.0

Detects **Tuberculosis**, **Pneumothorax**, **Pneumonia**, and **Normal** from chest X-rays.
Trained on single-label folders → predicts **multiple diseases simultaneously** at inference.

---

## 🆕 What's New in v2.0

| Feature | v1.0 | v2.0 |
|---|---|---|
| Model | EfficientNet-B0 | **EfficientNet-B2** |
| Image size | 224×224 | **260×260** |
| Dropout | 0.3 | **0.5** |
| Classifier head | Simple Linear | **BN → Dropout → Linear → SiLU → BN → Dropout → Linear** |
| Loss | BCEWithLogits | **BCEWithLogits + Label Smoothing (0.1)** |
| Learning rate | 1e-4 | **5e-5** |
| Weight decay | 1e-5 | **1e-4** |
| LR schedule | ReduceLROnPlateau | **CosineAnnealingLR** |
| Backbone unfreeze | Same LR | **Differential LR (backbone = LR/10)** |
| Augmentation | Basic | **+ Affine, GaussianBlur, Sharpness, RandomErasing** |
| Mixup | ❌ | **✅ alpha=0.3, 50% of batches** |
| Threshold | Single 0.45 | **Per-class tuned threshold** |
| Dataset split | Random | **Stratified per class** |
| Epochs / Patience | 40 / 7 | **60 / 10** |
| Checkpoint | Weights only | **Weights + thresholds + classes + image size** |
| Expected F1 | ~68% | **~85–92%** |

---

## 📁 Required Folder Structure

```
xray_project/
├── dataset/
│   ├── Tuberculosis/       ← ~2100 images
│   ├── Pneumothorax/       ← ~2100 images
│   ├── Pneumonia/          ← ~2100 images
│   └── Normal/             ← ~2100 images
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```

> ⚠️ Folder names must **exactly** match the `classes` list in `train.py`

---

## ⚙️ Setup

```bash
# Step 1 — Create virtual environment
python -m venv venv

# Step 2 — Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Step 3 — Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Training

```bash
python train.py
```

### What happens step by step:

**Epochs 1–5 (Warmup)**
- Only the classifier head trains
- Backbone (EfficientNet-B2) is frozen
- Safe, fast warm-up

**Epoch 6 onwards (Full Fine-tuning)**
- Backbone unfrozen with **differential learning rates**
- Backbone LR = 5e-6 (10× lower)
- Classifier LR = 5e-5
- Prevents destroying pretrained features

**After training completes**
- Per-class thresholds tuned on validation set
- Full evaluation on test set
- All results printed + saved

### What gets saved:

| File | Description |
|---|---|
| `best_model.pth` | Best model weights only |
| `best_model_full.pth` | Weights + thresholds + classes + image size |
| `training_curves.png` | Loss & F1 curves across epochs |
| `confusion_matrix.png` | Combined confusion matrix heatmap |

---

## 🔍 Inference — Single Image

```bash
# Basic prediction (prints results to terminal)
python inference.py --image path/to/xray.jpg

# With visualization saved as PNG
python inference.py --image path/to/xray.jpg --show

# Custom checkpoint
python inference.py --image path/to/xray.jpg --model best_model_full.pth
```

### Example terminal output:

```
==================================================
  PREDICTION RESULTS
==================================================
  Image : patient_001.jpg

  🔴 Tuberculosis      : 84.32%  (threshold: 0.40) ← DETECTED
  ⚪ Pneumothorax      : 11.20%  (threshold: 0.45)
  🔴 Pneumonia         : 71.89%  (threshold: 0.35) ← DETECTED
  ⚪ Normal            :  3.45%  (threshold: 0.50)

  ──────────────────────────────────────────────
  ⚠️  Disease(s) Detected: Tuberculosis, Pneumonia
==================================================
```

---

## 🔍 Inference — Batch (Multiple Images)

```python
from inference import predict_batch

results = predict_batch(["xray1.jpg", "xray2.png", "xray3.jpg"])

for path, prob_dict, predictions in results:
    print(f"{path}: {predictions}")
```

---

## 📊 What the Evaluation Shows

After training you will see:

```
  Metric                   Score         %
  ─────────────────────────────────────────
  Accuracy (exact match)   0.8750     87.50%
  Macro Precision          0.8921     89.21%
  Macro Recall             0.8734     87.34%
  Macro F1                 0.8826     88.26%
  Macro AUC-ROC            0.9412     94.12%

  PER-CLASS AUC-ROC
  Tuberculosis        : 0.9521  [████████████████████]  95.2%
  Pneumothorax        : 0.9387  [██████████████████░░]  93.9%
  Pneumonia           : 0.9214  [██████████████████░░]  92.1%
  Normal              : 0.9124  [██████████████████░░]  91.2%

  PER-CLASS TUNED THRESHOLDS
  Tuberculosis        : 0.40  [████████░░░░░░░░░░░░]
  Pneumothorax        : 0.45  [█████████░░░░░░░░░░░]
  Pneumonia           : 0.35  [███████░░░░░░░░░░░░░]
  Normal              : 0.50  [██████████░░░░░░░░░░]
```

---

## 🧠 How It Works — Key Concepts

### Single-label training → Multi-label inference
Each image in training has **one disease label** (from its folder). The model uses **BCEWithLogitsLoss** with a **sigmoid output per class** — meaning each class is scored independently. At inference, any class with probability ≥ its threshold is reported as detected, so **multiple diseases can fire simultaneously**.

### Label Smoothing
Instead of training with hard labels (0 or 1), targets are softened:
- `1 → 0.95` (positive label)
- `0 → 0.05` (negative label)

This prevents the model from becoming overconfident and reduces overfitting.

### Mixup Augmentation
50% of training batches mix two random images and their labels:
```
mixed_image = 0.7 × image_A + 0.3 × image_B
mixed_label = 0.7 × label_A + 0.3 × label_B
```
Forces the model to learn smoother decision boundaries.

### Differential Learning Rate
When the backbone unfreezes at epoch 6:
- Backbone gets **LR/10** → gentle updates to preserve ImageNet features
- Classifier head gets **full LR** → faster learning for new task

### Per-Class Threshold Tuning
After training, the best threshold for each class is found separately on the validation set. This is more accurate than using a single threshold for all classes because different diseases have different confidence distributions.

---

## ⚙️ Key Config Options (train.py)

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | `"dataset"` | Root folder of your data |
| `classes` | 4 diseases | Must match folder names exactly |
| `image_size` | `260` | EfficientNet-B2 native size |
| `batch_size` | `16` | Lower to `8` if RAM is limited |
| `num_epochs` | `60` | Max training epochs |
| `warmup_epochs` | `5` | Epochs before backbone unfreezes |
| `learning_rate` | `5e-5` | AdamW base learning rate |
| `weight_decay` | `1e-4` | L2 regularization strength |
| `early_stop_patience` | `10` | Stop if no improvement for N epochs |
| `label_smoothing` | `0.1` | Soft label strength |
| `mixup_alpha` | `0.3` | Mixup strength (0 = disable) |
| `mixup_prob` | `0.5` | Fraction of batches using mixup |
| `threshold` | `0.40` | Default threshold (tuned per-class after training) |

---

## 🗂️ Output Files Summary

| File | When Created | Contents |
|---|---|---|
| `best_model.pth` | During training | Best model weights |
| `best_model_full.pth` | After training | Weights + thresholds + metadata |
| `training_curves.png` | After training | Loss & F1 plots |
| `confusion_matrix.png` | After evaluation | 4×4 heatmap |
| `prediction_result.png` | After inference `--show` | X-ray + probability bars |

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `Folder not found` error | Check folder names match `classes` in CONFIG exactly |
| Out of memory | Lower `batch_size` to `8` |
| Training very slow | Normal for CPU — expect 8–15 min per epoch |
| Low Normal class F1 | Collect more diverse Normal images |
| Checkpoint not found | Run `train.py` first before `inference.py` |
| `venv\Scripts\activate` blocked | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first |
