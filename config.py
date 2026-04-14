import os
import torch

# ---------------------------------------------------------------------------
# Pre-trained HuggingFace model (Phase 1)
# ---------------------------------------------------------------------------
HF_MODEL_ID = "dima806/deepfake_vs_real_image_detection"

# ---------------------------------------------------------------------------
# Custom model checkpoint (Phase 2 — fill in after training)
# ---------------------------------------------------------------------------
CUSTOM_CHECKPOINT = "checkpoints/best_model.pth"

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Data paths (Phase 2 training)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
VAL_DIR = os.path.join(DATA_DIR, "Validation")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# ---------------------------------------------------------------------------
# Training hyperparameters (Phase 2)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
EARLY_STOPPING_PATIENCE = 5

# ---------------------------------------------------------------------------
# Normalization (ImageNet stats — used by both HF model and custom model)
# ---------------------------------------------------------------------------
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
