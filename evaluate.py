"""
Evaluation script — runs on a test split and prints metrics.

Supports both models:
    Phase 1 (HuggingFace pre-trained):
        python evaluate.py --model hf

    Phase 2 (custom checkpoint):
        python evaluate.py --model custom --checkpoint checkpoints/best_model.pth

    Side-by-side comparison (Phase 2):
        python evaluate.py --model both --checkpoint checkpoints/best_model.pth
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, auc, classification_report,
    confusion_matrix, roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

import config
from models.efficientnet import DeepfakeClassifier
from utils.dataset import DeepfakeDataset, get_transforms
from utils.face_detector import detect_and_crop


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["hf", "custom", "both"], default="hf")
    p.add_argument("--checkpoint", default=config.CUSTOM_CHECKPOINT)
    p.add_argument("--test_dir", default=config.VAL_DIR,
                   help="Directory with real/ and fake/ subfolders")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--no_face_crop", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_custom(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in tqdm(loader, desc="Custom model"):
        logits = model(images.to(device))
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())
    return np.array(all_labels), np.array(all_probs)


def predict_hf(test_dir: str, use_face_crop: bool):
    processor = AutoImageProcessor.from_pretrained(config.HF_MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(config.HF_MODEL_ID)
    model.eval().to(config.DEVICE)

    # HF model label mapping
    id2label = model.config.id2label  # e.g. {0: "Fake", 1: "Real"}
    fake_idx = next(k for k, v in id2label.items() if v.lower() == "fake")

    from utils.dataset import _collect_samples
    samples = _collect_samples(test_dir)

    all_probs, all_labels = [], []
    for path, label in tqdm(samples, desc="HF model"):
        image = Image.open(path).convert("RGB")
        if use_face_crop:
            image, _ = detect_and_crop(image)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        all_probs.append(probs[fake_idx].item())
        all_labels.append(label)

    return np.array(all_labels), np.array(all_probs)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    report = classification_report(labels, preds, target_names=["Real", "Fake"])
    cm = confusion_matrix(labels, preds)
    return {"acc": acc, "auc": roc_auc, "report": report, "cm": cm, "fpr": fpr, "tpr": tpr}


def print_metrics(name, m):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {m['acc']:.4f}")
    print(f"  AUC-ROC  : {m['auc']:.4f}")
    print(f"\n{m['report']}")


def save_confusion_matrix(cm, name, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"]); ax.set_yticklabels(["Real", "Fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


def save_roc_curves(results: dict[str, dict], out_path: str):
    plt.figure(figsize=(6, 5))
    for name, m in results.items():
        plt.plot(m["fpr"], m["tpr"], label=f"{name} (AUC={m['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    use_face_crop = not args.no_face_crop
    results = {}

    if args.model in ("hf", "both"):
        labels, probs = predict_hf(args.test_dir, use_face_crop)
        m = compute_metrics(labels, probs)
        print_metrics("HuggingFace Pre-trained", m)
        save_confusion_matrix(m["cm"], "HF", os.path.join(config.CHECKPOINT_DIR, "cm_hf.png"))
        results["HF Pre-trained"] = m

    if args.model in ("custom", "both"):
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            print("Custom checkpoint not found. Train first with python train.py")
            return

        test_ds = DeepfakeDataset(args.test_dir, split="val", use_face_crop=use_face_crop)
        loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=config.NUM_WORKERS)

        model = DeepfakeClassifier(pretrained=False)
        ckpt  = torch.load(args.checkpoint, map_location=config.DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(config.DEVICE)

        labels, probs = predict_custom(model, loader, config.DEVICE)
        m = compute_metrics(labels, probs)
        print_metrics("Custom EfficientNet-B0", m)
        save_confusion_matrix(m["cm"], "Custom", os.path.join(config.CHECKPOINT_DIR, "cm_custom.png"))
        results["Custom EfficientNet-B0"] = m

    if len(results) > 1:
        save_roc_curves(results, os.path.join(config.CHECKPOINT_DIR, "roc_comparison.png"))


if __name__ == "__main__":
    main()
