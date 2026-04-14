# Deployment Guide: Deepfake Detection System

This guide covers deploying the Deepfake Detection System to **Streamlit Cloud** (production) and running locally.

---

## 🚀 Option 1: Deploy to Streamlit Cloud (Recommended)

### Prerequisites
- GitHub account
- Streamlit Cloud account (free)
- Git installed locally

### Step 1: Initialize Git Repository

```bash
cd ~/Documents/MiniProj

# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Deepfake Detection System with EfficientNet-B0"
```

### Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Create a new public repository named `deepfake-detection`
3. Follow GitHub's instructions to push your code:

```bash
git remote add origin https://github.com/YOUR_USERNAME/deepfake-detection.git
git branch -M main
git push -u origin main
```

**Important:** Since model files (`.pth`) are large:
- Add to `.gitignore` (already done)
- Upload checkpoint files manually to Streamlit Cloud via Secrets, OR
- Document how to download them

### Step 3: Prepare for Cloud Deployment

**Download checkpoint and store in repo** (or reference external source):

For Streamlit Cloud, large model files should be:
1. Hosted externally (Hugging Face Hub, AWS S3, etc.)
2. Downloaded on first run
3. Cached locally

For now, we'll use the approach where models are pre-uploaded.

### Step 4: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click "New app"
3. Select your GitHub repository: `deepfake-detection`
4. Set main file path: `app.py`
5. Click "Deploy"

Streamlit will:
- Install dependencies from `requirements.txt`
- Start the app
- Generate a public URL (e.g., `https://deepfake-detection-<hash>.streamlit.app`)

### Step 5: Configure Secrets (Optional)

If you have API keys or sensitive config:
1. Click "Advanced settings"
2. Add secrets in `.streamlit/secrets.toml` format
3. Reference in code as: `st.secrets["key_name"]`

---

## 💻 Option 2: Run Locally (Development)

### Setup

```bash
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepfake

# Install dependencies (if needed)
pip install -r requirements.txt

# Run app
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## 📦 Managing Large Model Files

Since `.pth` files are ~19MB and Streamlit Cloud has limitations:

### Option A: Use Hugging Face Hub (Recommended)

```python
from huggingface_hub import hf_hub_download

# In app.py or config.py
model_path = hf_hub_download(
    repo_id="your-username/deepfake-detection",
    filename="efficientnet_best.pth"
)
```

### Option B: AWS S3 / Cloud Storage

```python
import boto3

s3 = boto3.client('s3')
s3.download_file('my-bucket', 'efficientnet_best.pth', 'checkpoints/efficientnet_best.pth')
```

### Option C: Include in Git LFS

```bash
# Install Git LFS
git lfs install

# Track .pth files
git lfs track "*.pth"
git add .gitattributes

# Commit and push
git add checkpoints/*.pth
git commit -m "Add model checkpoints"
git push
```

---

## 🔧 Environment Variables for Cloud

Create `.streamlit/secrets.toml` in repo:

```toml
# Hugging Face token (for downloading models)
HF_TOKEN = "your-huggingface-token"

# AWS credentials (if using S3)
AWS_ACCESS_KEY_ID = "your-key"
AWS_SECRET_ACCESS_KEY = "your-secret"

# Custom checkpoint path
CUSTOM_CHECKPOINT_URL = "https://your-bucket.s3.amazonaws.com/efficientnet_best.pth"
```

Access in code:
```python
import streamlit as st
hf_token = st.secrets.get("HF_TOKEN")
```

---

## ✅ Testing Before Deployment

### Local Testing

```bash
# Test all imports
python -c "from models.efficientnet import DeepfakeClassifier; print('✓')"

# Test app startup
streamlit run app.py --logger.level=debug

# Test with sample image
# 1. Upload an image in UI
# 2. Verify detection works
# 3. Check Grad-CAM heatmap
# 4. Verify XAI explanation (3-5 lines)
```

### Cloud Testing

1. Deploy to Streamlit Cloud
2. Wait for build to complete (2-5 minutes)
3. Access the public URL
4. Test with sample images
5. Check logs for errors

---

## 📊 Performance Optimization

### For Streamlit Cloud

```python
# app.py optimizations

# 1. Cache model loading (already done)
@st.cache_resource
def load_model():
    ...

# 2. Limit max upload size
# In .streamlit/config.toml:
# [server]
# maxUploadSize = 200  # MB

# 3. Reduce image quality if needed
image = image.resize((500, 500))  # Before face detection

# 4. Use lightweight transforms
from torchvision.transforms import v2
```

### Memory Management

Streamlit Cloud free tier: ~800MB RAM
- Model: ~50MB
- Cache: ~100MB
- Other: ~200MB
- Available for inference: ~450MB

**Safe limits:**
- Single inference: <100ms
- Batch size: 1 (no batching on free tier)

---

## 🐛 Troubleshooting

### Model Not Found
```
Error: checkpoints/efficientnet_best.pth not found
```
**Solution**: Upload model to cloud storage and download on init

### Out of Memory
```
MemoryError: Unable to allocate X.XX GiB for an array
```
**Solution**: Reduce batch size or upgrade to paid tier

### Long Load Time
```
Loading model... [takes >30 seconds]
```
**Solution**: Optimize imports, use cache, consider model quantization

### Import Errors
```
ModuleNotFoundError: No module named 'timm'
```
**Solution**: Ensure all dependencies in `requirements.txt`, run `pip install -r requirements.txt`

---

## 📋 Deployment Checklist

- [ ] All code committed to Git
- [ ] `.gitignore` configured (models excluded)
- [ ] `requirements.txt` updated and tested
- [ ] `.streamlit/config.toml` created
- [ ] Model files accessible (local/cloud)
- [ ] App tested locally (`streamlit run app.py`)
- [ ] GitHub repo created and code pushed
- [ ] Streamlit Cloud account created
- [ ] App deployed to Streamlit Cloud
- [ ] Public URL verified and working
- [ ] XAI explanations tested (3-5 lines)
- [ ] Inference tested with sample images

---

## 🎉 Post-Deployment

### Monitor Performance
- Check Streamlit Cloud logs
- Monitor inference speed
- Track errors and usage

### Updates
```bash
# Make code changes locally
git add .
git commit -m "Update: feature description"
git push origin main

# Streamlit Cloud auto-redeploys on push
```

### Scaling
- Free tier: Limited (restart after 1 hour inactivity)
- Pro tier: Persistent, better resources
- Business tier: Custom infrastructure

---

## 📞 Support

- **Streamlit Docs**: https://docs.streamlit.io
- **GitHub Issues**: Create issue in your repo
- **Streamlit Community**: https://discuss.streamlit.io

---

## 🔗 Useful Links

- Streamlit Cloud: https://share.streamlit.io
- Streamlit Docs: https://docs.streamlit.io
- Hugging Face Hub: https://huggingface.co
- GitHub: https://github.com

---

**Last Updated**: 2026-04-14
**Deepfake Detection System v1.0**
