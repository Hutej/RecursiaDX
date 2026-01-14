# Offline Mode Architecture

## Overview

The GigaPath WSI Breast Cancer Lesion Analysis system is architected to support **fully offline operation** for core inference workflows, making it suitable for deployment in air-gapped research environments, secure medical facilities, and locations with limited internet connectivity.

---

## 🔒 Offline Capabilities

### Fully Offline Components

The following components operate **100% offline** after initial setup:

#### 1. **Preprocessing Pipeline**
- ✅ WSI tile extraction (`scripts/preprocess.py`)
- ✅ Tissue detection (Otsu thresholding, no external dependencies)
- ✅ Coordinate generation

#### 2. **Feature Extraction**
- ✅ Pre-trained model inference (`scripts/extract_features.py`)
- ✅ HDF5 caching
- ✅ Batch processing

**Requirement**: Model weights must be downloaded during initial setup.

#### 3. **Top-K Sampling**
- ✅ Feature ranking (`scripts/sample_tiles.py`)
- ✅ Tile selection
- ✅ Metadata generation

#### 4. **MIL Training**
- ✅ Model training (`scripts/train_mil.py`)
- ✅ Checkpoint saving
- ✅ Validation

#### 5. **Inference**
- ✅ Slide-level prediction (`scripts/infer_mil.py`)
- ✅ Confidence score calculation
- ✅ CSV output generation

#### 6. **Explainability**
- ✅ Attention heatmap generation (`scripts/generate_heatmaps.py`)
- ✅ Overlay creation
- ✅ Top-tile visualization

---

## 🌐 Internet-Dependent Components

### One-Time Setup (Internet Required)

The following require internet access **ONLY during initial installation**:

1. **Python Package Installation**
   ```bash
   pip install -r requirements.txt
   ```
   - PyTorch, torchvision
   - OpenSlide, h5py, pandas
   - scikit-learn, matplotlib

2. **Pre-trained Model Weights**
   - GigaPath backbone weights (if using)
   - Downloaded automatically by `torchvision` or Hugging Face Hub
   - **Solution**: Cache weights locally in `~/.cache/torch` or custom directory

### Optional Online Features

The following features **may** require internet but are optional:

1. **TensorBoard Logging** (if using remote server)
2. **Hugging Face Model Hub** (if downloading models dynamically)
3. **Progressive WSI Viewer** (if using web-based viewer - architecture only)

---

## 📦 Offline Deployment Guide

### Step 1: Online Machine Setup

On a machine **with internet access**:

```bash
# 1. Clone repository
git clone https://github.com/YourUsername/GigaPath-AI.git
cd GigaPath-AI

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\\venv\\Scripts\\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Cache model weights (if needed)
python -c "import torch; import torchvision; torchvision.models.resnet50(pretrained=True)"

# 5. Verify setup
python scripts/verify_setup.py
```

### Step 2: Transfer to Offline Machine

Package the entire environment:

```bash
# Option A: Archive entire venv + code
tar -czf gigapath-offline.tar.gz GigaPath-AI/

# Option B: Export pip dependencies for offline install
pip download -r requirements.txt -d packages/
```

Transfer to offline machine via:
- USB drive
- Secure file transfer
- Physical media

### Step 3: Offline Machine Activation

On the **offline machine**:

```bash
# Extract
tar -xzf gigapath-offline.tar.gz
cd GigaPath-AI

# Activate environment
source venv/bin/activate  # Linux/Mac
# OR
.\\venv\\Scripts\\activate  # Windows

# Verify (no internet required)
python scripts/verify_setup.py
```

---

## 🔍 Verifying Offline Operation

### Test Offline Mode

```bash
# Disable network (simulated)
# Linux: sudo ifconfig eth0 down
# Windows: disable network adapter

# Run full pipeline
python scripts/preprocess.py --input data/raw_wsi/test.svs --output data/tiles
python scripts/extract_features.py --input data/tiles --output data/features
python scripts/sample_tiles.py --input data/features --output data/features_topk
python scripts/infer_mil.py --model checkpoints/best_model.pth --features data/features_topk --output results/predictions.csv

# Re-enable network
```

If all steps complete without internet errors → **✅ Offline ready**

---

## ⚠️ Limitations in Offline Mode

### What Does NOT Work Offline

1. **Model Weight Downloads**
   - First-time model loading from Hugging Face Hub
   - **Workaround**: Pre-cache on online machine

2. **pip install** (obviously)
   - Cannot install new packages
   - **Workaround**: Freeze environment before transfer

3. **Online Documentation**
   - Cannot access GitHub issues, wikis
   - **Workaround**: Download docs as PDF/markdown

4. **Progressive WSI Viewer** (if web-based)
   - May require CDN resources (e.g., OpenSeadragon, Leaflet)
   - **Workaround**: Use cached/bundled viewer or disable

### What DOES Work Offline

✅ All core inference pipelines  
✅ Model training and evaluation  
✅ Heatmap generation  
✅ CSV/JSON output generation  
✅ Logging and checkpointing

---

## 🏗️ Architecture Considerations

### Model Weight Caching

**PyTorch Default Cache**: `~/.cache/torch/hub/checkpoints/`

**Custom Cache** (recommended):
```python
# In backbones.py
torch.hub.set_dir('/path/to/offline/cache')
```

### Data Storage

All data is local:
- `data/`: Raw WSI, tiles, features
- `checkpoints/`: Model weights
- `results/`: Predictions, visualizations
- `logs/`: Execution logs

**No external databases or cloud storage required.**

### Configuration

All configuration in `configs/config.yaml`:
- No API keys
- No external service URLs
- No telemetry or analytics

---

## 🚀 Production Deployment Scenarios

### Scenario 1: Air-Gapped Research Lab

**Environment**: Secure facility with no internet access

**Setup**:
1. Prepare on online machine
2. Transfer via secure media
3. Deploy to air-gapped workstation
4. Run full pipeline offline

**Use Case**: Sensitive patient data, classified research

---

### Scenario 2: Remote Field Site

**Environment**: Rural hospital with unreliable internet

**Setup**:
1. Pre-install on laptop with downloaded models
2. Process slides locally
3. Generate predictions offline
4. Sync results when connectivity available (optional)

**Use Case**: Resource-limited settings, disaster response

---

### Scenario 3: Secure Medical Facility

**Environment**: Hospital network isolated from internet

**Setup**:
1. Install on internal server
2. Integrate with PACS (optional)
3. Run predictions on-premises
4. Export results to local EM

R (optional)

**Use Case**: HIPAA-compliant environments, regulatory compliance

---

## 📚 Offline-First Design Principles

1. **No Cloud Dependencies**: All computation on-premises
2. **Local Storage**: No S3, Azure Blob, or cloud databases
3. **Pre-cached Models**: Weights bundled or cached locally
4. **Self-Contained**: Python environment + code = complete system
5. **Portable**: Can run on workstation, server, or HPC cluster

---

## ✅ Offline Readiness Checklist

Before deploying offline, ensure:

- [ ] All Python packages installed in venv
- [ ] Model weights cached locally
- [ ] Test data available
- [ ] Configuration file prepared
- [ ] Logs directory writable
- [ ] Sufficient disk space (see system requirements)
- [ ] GPU drivers installed (if using GPU)
- [ ] Full pipeline tested on online machine

---

## 🐛 Troubleshooting Offline Issues

### Issue: "ConnectionError" during model loading

**Cause**: Model weights not cached

**Solution**:
```bash
# On online machine
python -c "from src.feature_extraction import FeatureExtractor; FeatureExtractor('resnet50-imagenet')"
```

### Issue: "pip install" fails

**Cause**: Already offline

**Solution**: Transfer pre-downloaded packages:
```bash
# On online machine
pip download -r requirements.txt -d packages/

# On offline machine
pip install --no-index --find-links=packages/ -r requirements.txt
```

---

**Last Updated**: 2026-01-12

**Version**: 1.0
