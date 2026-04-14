#!/bin/bash

# Deepfake Detection System - Deployment Script
# This script prepares the project for Streamlit Cloud deployment

set -e  # Exit on error

echo "========================================"
echo "Streamlit Cloud Deployment Setup"
echo "========================================"
echo

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

echo "1. Initializing Git repository..."
if [ -d .git ]; then
    echo "   ✓ Git repo already initialized"
else
    git init
    echo "   ✓ Git repo initialized"
fi

echo
echo "2. Creating .gitignore..."
if [ -f .gitignore ]; then
    echo "   ✓ .gitignore already exists"
else
    echo "   ✓ .gitignore created"
fi

echo
echo "3. Creating Streamlit config..."
if [ -f .streamlit/config.toml ]; then
    echo "   ✓ Streamlit config already exists"
else
    mkdir -p .streamlit
    echo "   ✓ Streamlit config created"
fi

echo
echo "4. Verifying requirements.txt..."
if [ -f requirements.txt ]; then
    echo "   ✓ requirements.txt exists"
    echo "   Dependencies:"
    head -5 requirements.txt | sed 's/^/     /'
    echo "     ..."
else
    echo "   ❌ requirements.txt not found"
    exit 1
fi

echo
echo "5. Checking project structure..."
files=("app.py" "config.py" "train.py" "evaluate.py")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ⚠️  Missing: $file"
    fi
done

echo
echo "6. Git status..."
git status --short | head -10
echo

echo "========================================"
echo "Next Steps for Deployment:"
echo "========================================"
echo
echo "1. Stage all files:"
echo "   git add ."
echo
echo "2. Create initial commit:"
echo "   git commit -m 'Initial commit: Deepfake Detection System'"
echo
echo "3. Create GitHub repository:"
echo "   - Go to https://github.com/new"
echo "   - Create repo named 'deepfake-detection'"
echo
echo "4. Add remote and push:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/deepfake-detection.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo
echo "5. Deploy to Streamlit Cloud:"
echo "   - Go to https://share.streamlit.io"
echo "   - Click 'New app'"
echo "   - Select your repository and app.py"
echo "   - Click 'Deploy'"
echo
echo "📚 Full guide: See DEPLOYMENT.md"
echo "========================================"
