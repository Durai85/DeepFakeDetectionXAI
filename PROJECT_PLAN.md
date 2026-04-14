# Deepfake Detection — XAI Project Plan

## Project Summary

An **Explainable AI (XAI) framework** for deepfake image detection built as a lightweight
**Streamlit web application**. Users upload a face image; the system classifies it as
**REAL or FAKE** and generates a **Grad-CAM heatmap** that visually highlights which facial
regions (e.g., eye reflections, jawline, skin boundaries) drove the model's decision.

**Core innovation:** Bridging high-performance deep learning with human-understandable
forensic evidence through Grad-CAM visualisations — not just a binary result.

---

## Architecture Overview

```
User uploads image
        │
        ▼
MTCNN Face Detection  ──(fallback: center-crop)──►  224×224 Face Crop
        │
        ▼
EfficientNet-B0 Classifier  ──►  Logit  ──►  Sigmoid  ──►  Fake Probability
        │                                                         │
        ▼                                                         ▼
Grad-CAM (hooks on last                              REAL / FAKE badge
conv block)                                          Confidence bar
        │                                            XAI explanation text
        ▼
Heatmap overlay (JET colormap)
```

**Tech stack:**
| Component        | Choice                                          |
|------------------|-------------------------------------------------|
| Language         | Python 3.9+                                     |
| ML Framework     | PyTorch + `timm`                                |
| Model backbone   | EfficientNet-B0 (ImageNet pre-trained)          |
| Face detection   | MTCNN via `facenet-pytorch`                     |
| XAI method       | Grad-CAM (hook-based, no external library)      |
| Web app          | Streamlit                                       |
| HF baseline      | `dima806/deepfake_vs_real_image_detection`      |
| Augmentation     | RandomFlip, ColorJitter, GaussianBlur, Rotation |

---

## Two-Phase Approach

### Phase 1 — Pre-trained HuggingFace Baseline (COMPLETE — runnable now)
- Uses `dima806/deepfake_vs_real_image_detection` from HuggingFace Hub.
- No training required; weights download automatically on first run.
- Grad-CAM hooks onto `model.efficientnet.encoder.blocks[-1]`.
- **How to run:** `streamlit run app.py` → select "Pre-trained (HuggingFace)" in sidebar.

### Phase 2 — Custom Fine-tuned Model (IN PROGRESS — needs data)
- Fine-tune EfficientNet-B0 on a local dataset (`data/train/real`, `data/train/fake`).
- Save best checkpoint to `checkpoints/best_model.pth`.
- Load in the app via "Custom (load checkpoint)" in sidebar.
- **How to run:** `python train.py` then `python evaluate.py --model custom --checkpoint checkpoints/best_model.pth`

---

## File Structure

```
MiniProj/
├── app.py                  # Streamlit web app (Phase 1 + 2)
├── train.py                # Phase 2 training script
├── evaluate.py             # Evaluation: accuracy, AUC-ROC, confusion matrix
├── config.py               # All paths and hyperparameters
├── requirements.txt        # Python dependencies
│
├── models/
│   ├── __init__.py
│   └── efficientnet.py     # DeepfakeClassifier (EfficientNet-B0 + custom head)
│
├── utils/
│   ├── __init__.py
│   ├── dataset.py          # DeepfakeDataset (real/fake folders → PyTorch Dataset)
│   ├── face_detector.py    # MTCNN detect_and_crop + draw_box
│   └── gradcam.py          # GradCAM class + overlay_heatmap + get_target_layer_hf
│
├── data/
│   ├── train/
│   │   ├── real/           # ← EMPTY: place training real images here
│   │   └── fake/           # ← EMPTY: place training fake images here
│   └── val/
│       ├── real/           # ← EMPTY: place validation real images here
│       └── fake/           # ← EMPTY: place validation fake images here
│
├── checkpoints/            # ← EMPTY: best_model.pth saved here after training
│
├── PROJECT_PLAN.md         # This file
└── DEEPFAKE_DETECTION_PROJECT.md  # Original project specification document
```

---

## What Is Already Done

| Component | File | Status |
|-----------|------|--------|
| Streamlit app with 3-column layout | `app.py` | Done |
| HuggingFace model loading + inference | `app.py` | Done |
| Custom checkpoint loading + inference | `app.py` | Done |
| Grad-CAM heatmap generation + overlay | `utils/gradcam.py` | Done |
| MTCNN face detection with fallback | `utils/face_detector.py` | Done |
| EfficientNet-B0 classifier + custom head | `models/efficientnet.py` | Done |
| Dataset loader with augmentation | `utils/dataset.py` | Done |
| Training loop with early stopping | `train.py` | Done |
| Evaluation: AUC, accuracy, confusion matrix, ROC curves | `evaluate.py` | Done |
| Config with all hyperparameters | `config.py` | Done |
| XAI explanation text (per prediction) | `app.py` | Done |
| REAL/FAKE badge + confidence progress bar | `app.py` | Done |

---

## What Is Left To Do

### Priority 1 — Verify Phase 1 works (quick win)
- [ ] Run `streamlit run app.py`
- [ ] Upload a test image (any face photo)
- [ ] Confirm face detection, REAL/FAKE badge, Grad-CAM heatmap, and explanation text
      all render correctly
- [ ] Check no import errors or missing dependencies

### Priority 2 — Collect data for Phase 2 custom training
Data directories `data/train/real`, `data/train/fake`, `data/val/real`, `data/val/fake`
are all empty. Populate them before running `train.py`.

**Recommended datasets (easiest to hardest to obtain):**

| Dataset | Images | Where to get | Use for |
|---------|--------|--------------|---------|
| **CIFake** | 60k real + 60k fake | Kaggle (`jordanharris/cifake-real-and-ai-generated-synthetic-images`) | Quick start — balanced, easy download |
| **140k Real vs Fake** | 70k real + 70k fake | Kaggle (`xhlulu/140k-real-and-fake-faces`) | Face-focused, good quality |
| **FaceForensics++** | Variable | [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) — requires form | Academic standard benchmark |
| **Celeb-DF (v2)** | 5,639 clips → extract frames | [github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics) | High-quality fakes, fewer obvious artifacts |

**Minimum to start training:** ~2,000 images per class (real + fake) in train, ~500 per class in val.

**Data split:** 70% train / 15% val / 15% test (use val as test for now).

**Folder structure expected by `DeepfakeDataset`:**
```
data/train/real/*.jpg
data/train/fake/*.jpg
data/val/real/*.jpg
data/val/fake/*.jpg
```

### Priority 3 — Run training (after data is in place)
```bash
python train.py
# or with custom args:
python train.py --epochs 30 --batch_size 16 --lr 1e-4
```
- Best checkpoint auto-saved to `checkpoints/best_model.pth`
- Training curves saved to `checkpoints/training_curves.png`
- Early stopping patience: 5 epochs (set in `config.py`)

**Key hyperparameters (all in `config.py`):**
| Param | Default | Notes |
|-------|---------|-------|
| `EPOCHS` | 20 | Increase to 30 for better convergence |
| `BATCH_SIZE` | 32 | Reduce to 16 if GPU OOM |
| `LR` | 1e-4 | Adam optimizer |
| `EARLY_STOPPING_PATIENCE` | 5 | Epochs without val improvement before stopping |
| `IMG_SIZE` | 224 | Input resolution |

### Priority 4 — Evaluate the custom model
```bash
# Evaluate HF baseline only
python evaluate.py --model hf

# Evaluate custom model
python evaluate.py --model custom --checkpoint checkpoints/best_model.pth

# Side-by-side comparison
python evaluate.py --model both --checkpoint checkpoints/best_model.pth
```
Outputs saved to `checkpoints/`:
- `cm_hf.png` — confusion matrix for HF model
- `cm_custom.png` — confusion matrix for custom model
- `roc_comparison.png` — ROC curve overlay

**Target metrics:**
- Accuracy > 90% on validation set
- AUC-ROC > 0.95

### Priority 5 (Optional) — Add MesoNet backbone
The project spec mentions MesoNet as an alternative to EfficientNet. If needed, add
`models/mesonet.py` with `MesoNet4` and `MesoInception4` classes. These are small
custom CNNs designed specifically for deepfake detection (not ImageNet pre-trained).
Useful for comparing a domain-specific architecture vs a transfer learning approach.

---

## How to Run (Quick Reference)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app (Phase 1 — no training needed)
streamlit run app.py

# Train custom model (Phase 2 — requires data in data/train and data/val)
python train.py

# Evaluate
python evaluate.py --model hf                                       # HF baseline
python evaluate.py --model custom --checkpoint checkpoints/best_model.pth
python evaluate.py --model both   --checkpoint checkpoints/best_model.pth
```

---

## Key Design Decisions

1. **Grad-CAM without external libraries** — implemented from scratch in `utils/gradcam.py`
   using PyTorch forward/backward hooks. Avoids version compatibility issues with
   `pytorch-gradcam` or `captum`.

2. **Two-mode app** — Phase 1 (HF model) lets the app run immediately with no training.
   Phase 2 (custom checkpoint) is unlocked after training. Both use the same Grad-CAM
   pipeline and UI.

3. **MTCNN with center-crop fallback** — if no face is detected, the app still runs on a
   center-crop and shows a warning. Prevents hard crashes on non-face images.

4. **BCEWithLogitsLoss + single logit output** — numerically stable binary classification.
   Sigmoid is only applied at inference time.

5. **EfficientNet-B0 over B3** — B0 is faster and sufficient for a lightweight Streamlit
   demo. B3 could be used if higher accuracy is needed at the cost of inference speed.

---

## Known Issues / Watched Items

- `data/train/` and `data/val/` are currently empty — Phase 2 cannot run until populated.
- `config.CUSTOM_CHECKPOINT = None` — must be updated to `"checkpoints/best_model.pth"`
  after training, or set it via the Streamlit sidebar at runtime.
- Grad-CAM for the HF model depends on the internal structure of
  `dima806/deepfake_vs_real_image_detection`. If HuggingFace updates the model weights
  or architecture, `get_target_layer_hf()` in `utils/gradcam.py` may need updating.
