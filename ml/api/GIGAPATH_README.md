# GigaPath Cancer Detection - Inference Setup

## Overview

This document explains the inference-only setup for tissue biopsy cancer detection using the GigaPath MIL model.

---

## ⚠️ Important: No Dataset Required

This is an **INFERENCE-ONLY** setup. You do NOT need:
- Training datasets
- Raw WSI files
- AWS credentials
- Re-training the model

All you need is the pre-trained checkpoint: `best_model.pth`

---

## Quick Start

### 1. Prerequisites

```bash
# Install dependencies (if not already installed)
pip install torch torchvision flask flask-cors pillow numpy h5py
```

### 2. Place Checkpoint

Ensure `best_model.pth` is at:
```
GigaPath-AI-WSI-Breast-Cancer-Lesion-Analysis/checkpoints/best_model.pth
```

### 3. Start GigaPath API

```bash
cd ml/api
python gigapath_api.py --port 5002
```

### 4. Start Main ML API

```bash
cd ml/api
python app.py
```

The main API (port 5000) will automatically proxy tissue requests to GigaPath API (port 5002).

---

## API Endpoints

### GigaPath API (Port 5002)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single image prediction |
| `/batch_predict` | POST | Multiple images |
| `/model_info` | GET | Model information |

### Main API (Port 5000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Routes to GigaPath (tissue) or Malaria/Platelet (blood) |
| `/batch_predict` | POST | Batch predictions |
| `/health` | GET | Health check |

---

## Request Format

### Single Prediction

```bash
curl -X POST http://localhost:5002/predict \
  -F "image=@/path/to/tissue_biopsy.png"
```

### Response

```json
{
  "success": true,
  "prediction": {
    "class": "Tumor",
    "is_tumor": true,
    "confidence": 0.92,
    "probability": 0.92,
    "risk_level": "High Risk"
  },
  "probabilities": {
    "normal": 0.08,
    "tumor": 0.92
  },
  "metadata": {
    "model": "GigaPath-AttentionMIL",
    "processing_time_ms": 245.5
  }
}
```

---

## Troubleshooting

### Checkpoint Not Found

```
TRAINED CHECKPOINT NOT FOUND
Expected checkpoint at: /path/to/best_model.pth
```

**Solution**: Place `best_model.pth` in the checkpoints directory.

### GigaPath API Not Available

```
GigaPath API is not available
Please start the GigaPath API: python gigapath_api.py --port 5002
```

**Solution**: Start the GigaPath API on port 5002.

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Backend       │
│   (React)       │     │   (Node.js)     │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │  ML API (5000) │
                        └────────┬───────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
     ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
     │ GigaPath (5002) │ │ Malaria Model   │ │ Platelet Model  │
     │ (Tissue Biopsy) │ │ (Blood Smear)   │ │ (Blood Smear)   │
     └─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## Model Details

| Property | Value |
|----------|-------|
| Model | AttentionMIL (Gated Attention) |
| Backbone | ResNet50 (ImageNet) |
| Feature Dim | 2048 |
| Input | 224×224 (auto-resized) |
| Output | Binary (Tumor/Normal) |
| Checkpoint Size | ~27 MB |

---

*Last Updated: January 2026*
