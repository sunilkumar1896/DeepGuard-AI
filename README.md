# AI-Powered Deepfake & Media Verification Platform

> **Hybrid CNN–Transformer architecture** for detecting deepfakes in images and videos — trained entirely on our own models, no external detection APIs.

---

## 🎯 Project Overview

This platform detects AI-generated or manipulated media using a novel **Hybrid CNN–Transformer** deep learning architecture that combines:

- **Local feature extraction** (CNN) — blending artifacts, texture inconsistencies, boundary issues
- **Global relationship modeling** (Transformer) — spatial and temporal coherence across facial regions and video frames

### What it can analyze
| Input | Output |
|-------|--------|
| Images (JPG, PNG) | REAL / FAKE + confidence score + Grad-CAM heatmap |
| Videos (MP4, AVI) | Frame-level analysis + suspicious frame detection + temporal confidence |
| Webcam (live) | Real-time classification |

---

## 🏗️ Architecture

```
Input (Image / Video Frame)
         ↓
   Face Detection & Crop
         ↓
   CNN Feature Extractor          ← EfficientNet-B4 (local artifacts)
         ↓
   Feature Embeddings [B, T, D]
         ↓
   Transformer Encoder            ← Multi-head attention (global/temporal)
         ↓
   Classification Head
         ↓
   REAL / DEEPFAKE + Confidence
```

### Three Models Built & Compared
1. **CNN Baseline** — EfficientNet-B4 fine-tuned for deepfake detection
2. **Transformer Baseline** — Vision Transformer (ViT) for global feature modeling
3. **Hybrid CNN-Transformer** *(main model)* — Combined architecture with temporal attention for video

---

## 📁 Project Structure

```
deepfake_platform/
├── configs/            ← config.yaml — all hyperparameters
├── data/
│   ├── raw/            ← FaceForensics++, DFDC (download separately)
│   ├── processed/      ← Cropped faces
│   └── splits/         ← Train/val/test JSON manifests
├── preprocessing/      ← Video, face detection, dataset pipelines
├── models/             ← CNN, Transformer, Hybrid architectures
├── training/           ← Training loops with checkpointing
├── evaluation/         ← Metrics, confusion matrix, AUC plots
├── inference/          ← Image & video predictor
├── explainability/     ← Grad-CAM heatmaps
├── backend/            ← FastAPI REST API
├── frontend/           ← React web application
├── scripts/            ← Setup, dataset download helpers
├── notebooks/          ← Exploratory analysis
├── results/            ← Evaluation plots, reports
└── requirements.txt
```

---

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
# Detect GPU and install correct PyTorch build automatically:
python scripts/setup_environment.py

# OR manually (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 2 — Verify environment
```bash
python scripts/check_environment.py
```

### Step 3 — Prepare dataset
```bash
# Organize your downloaded FaceForensics++ data
python preprocessing/dataset_preparation.py --dataset faceforensics --input data/raw/faceforensics

# Process faces
python preprocessing/face_detector.py --input data/raw --output data/processed

# Create train/val/test splits
python preprocessing/create_splits.py
```

### Step 4 — Train models
```bash
# CNN baseline
python training/train_cnn.py --config configs/config.yaml

# Transformer baseline
python training/train_transformer.py --config configs/config.yaml

# Hybrid (main model)
python training/train_hybrid.py --config configs/config.yaml
```

### Step 5 — Evaluate
```bash
python evaluation/evaluate.py --model hybrid --checkpoint checkpoints/hybrid/best_model.pth
```

### Step 6 — Run the application
```bash
# Backend (terminal 1)
cd backend
uvicorn main:app --reload --port 8000

# Frontend (terminal 2)
cd frontend
npm install && npm run dev
```

Open `http://localhost:5173` in your browser.

---

## 📊 Datasets

| Dataset | Size | Manipulation Types |
|---------|------|-------------------|
| [FaceForensics++](https://github.com/ondyari/FaceForensics) | ~1.5 TB (full) | Deepfakes, Face2Face, FaceSwap, NeuralTextures |
| [DFDC](https://ai.facebook.com/datasets/dfdc/) | ~470 GB | Various GAN-based methods |

> **Note**: Download datasets separately and place in `data/raw/faceforensics/` and `data/raw/dfdc/` respectively. See `scripts/dataset_download_guide.md` for links and instructions.

---

## 📈 Evaluation Metrics

All three models evaluated on:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC curve
- Confusion matrix
- Cross-dataset generalization (train on FF++, test on DFDC)
- Inference FPS and processing time

---

## 🔌 API Documentation

The FastAPI backend exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/analyze/image` | POST | Analyze uploaded image |
| `/api/v1/analyze/video` | POST | Analyze uploaded video |
| `/api/v1/history` | GET | Fetch analysis history |
| `/api/v1/history/{id}` | DELETE | Delete a record |
| `/api/v1/health` | GET | Health check |
| `/docs` | GET | Interactive Swagger UI |

---

## ⚙️ Configuration

All important parameters in `configs/config.yaml`:

```yaml
training:
  batch_size: 16
  learning_rate: 1.0e-4
  num_epochs: 50
  optimizer: "adamw"

model:
  cnn_backbone: "efficientnet_b4"

data:
  image_size: 224
  frames_per_video: 32
  augmentation_strength: "medium"
```

---

## 🔬 Research Contribution

This project demonstrates that a **Hybrid CNN–Transformer** architecture outperforms either architecture alone for deepfake detection by:

1. Extracting fine-grained local artifacts via CNN
2. Modeling long-range dependencies between facial regions via Transformer attention
3. Extending to temporal modeling for video sequences

---

## 📋 Requirements

- Python 3.9+
- PyTorch 2.1+
- CUDA 11.8+ (strongly recommended; CPU fallback available)
- 16 GB RAM minimum (32 GB recommended)
- 50 GB free disk space (for datasets and models)

---

## 📄 License

For academic/research use. See LICENSE file.

---

## 👥 Authors

4th Year AI/ML Research Project
