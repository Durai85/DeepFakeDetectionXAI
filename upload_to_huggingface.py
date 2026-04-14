#!/usr/bin/env python3
"""
Upload model checkpoints to Hugging Face Hub
Run once: python upload_to_huggingface.py
"""

from huggingface_hub import HfApi, create_repo
import os

# Configuration
HF_USERNAME = "Durai29"
REPO_NAME = "deepfake-models"
HF_REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

# Model files to upload
MODELS = [
    ("checkpoints/efficientnet_best.pth", "96.63% accuracy - Epoch 12"),
    ("checkpoints/best_model.pth", "91.98% accuracy - Epoch 3"),
]

print("=" * 70)
print("Hugging Face Model Upload Tool")
print("=" * 70)
print()

# Initialize API
api = HfApi()

print(f"Target repository: {HF_REPO_ID}")
print()

# Create repo if it doesn't exist
print("1. Creating/connecting to repository...")
try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        private=False,
        exist_ok=True
    )
    print(f"   ✓ Repository ready: https://huggingface.co/{HF_REPO_ID}")
except Exception as e:
    print(f"   ✓ Repository already exists")

print()
print("2. Uploading model files...")

# Upload files
for model_path, description in MODELS:
    if not os.path.exists(model_path):
        print(f"   ✗ {model_path} not found - skipping")
        continue

    filename = os.path.basename(model_path)
    filesize_mb = os.path.getsize(model_path) / (1024 * 1024)

    print(f"   Uploading {filename} ({filesize_mb:.1f} MB)...")
    print(f"   Description: {description}")

    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            repo_type="model"
        )
        print(f"   ✓ {filename} uploaded successfully")
    except Exception as e:
        print(f"   ✗ Error uploading {filename}: {e}")

    print()

print("=" * 70)
print("Upload Complete!")
print("=" * 70)
print()
print("Your models are now available at:")
print(f"  https://huggingface.co/{HF_REPO_ID}")
print()
print("The app will now automatically download them from Hugging Face.")
print()
