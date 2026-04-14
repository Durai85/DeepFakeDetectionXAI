"""
Phase 2 training script — custom EfficientNet-B0 deepfake classifier.

Usage
-----
    python train.py
    python train.py --train_dir data/train --val_dir data/val --epochs 30

The best model (lowest val loss) is saved to checkpoints/best_model.pth.
Training curves are saved to checkpoints/training_curves.png.
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models.efficientnet import DeepfakeClassifier
from utils.dataset import DeepfakeDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", default=config.TRAIN_DIR)
    p.add_argument("--val_dir", default=config.VAL_DIR)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--no_face_crop", action="store_true")
    p.add_argument("--save_name", default="best_model.pth",
                   help="Checkpoint filename saved under checkpoints/")
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def save_curves(train_losses, val_losses, train_accs, val_accs, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train")
    ax1.plot(epochs, val_losses, label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train")
    ax2.plot(epochs, val_accs, label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    use_face_crop = not args.no_face_crop
    train_ds = DeepfakeDataset(args.train_dir, split="train", use_face_crop=use_face_crop)
    val_ds   = DeepfakeDataset(args.val_dir,   split="val",   use_face_crop=use_face_crop)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True)

    device = config.DEVICE
    model = DeepfakeClassifier(pretrained=True).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print(f"Training on {device} | {len(train_ds)} train / {len(val_ds)} val samples")
    print(f"Checkpoint dir: {config.CHECKPOINT_DIR}\n")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        improved = vl_loss < best_val_loss
        marker = " *" if improved else ""
        print(
            f"Epoch [{epoch:02d}/{args.epochs}]  "
            f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}  |  "
            f"Val loss: {vl_loss:.4f}  acc: {vl_acc:.4f}{marker}"
        )

        if improved:
            best_val_loss = vl_loss
            patience_counter = 0
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, args.save_name)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": vl_loss, "val_acc": vl_acc}, ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    save_curves(train_losses, val_losses, train_accs, val_accs, config.CHECKPOINT_DIR)
    print("\nTraining complete. Curves saved to checkpoints/training_curves.png")


if __name__ == "__main__":
    main()
