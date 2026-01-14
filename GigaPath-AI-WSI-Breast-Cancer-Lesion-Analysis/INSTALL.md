# Installation Guide for Phase 0.1

## Quick Installation

### 1. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### 3. Install OpenSlide (System-Level)

**Windows**:
1. Download OpenSlide binaries: https://openslide.org/download/
2. Extract to `C:\OpenSlide`
3. Add to PATH: `C:\OpenSlide\bin`
4. Install Python bindings: `pip install openslide-python`

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get install openslide-tools
sudo apt-get install python3-openslide
pip install openslide-python
```

**macOS**:
```bash
brew install openslide
pip install openslide-python
```

### 4. Verify GPU (CUDA)
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**If CUDA not found**:
- Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Install PyTorch with CUDA support:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

### 5. Verify Installation
```bash
python scripts/verify_setup.py
```

---

## Expected Output (Successful Verification)

```
============================================================
  GigaPath WSI Pipeline - Phase 0.1 Verification
============================================================

============================================================
  Python Version Check
============================================================
Python version: 3.10.x
✓ Python version OK (3.8+)

============================================================
  Dependency Check
============================================================
✓ PyTorch installed
✓ TorchVision installed
✓ NumPy installed
✓ PyYAML installed
✓ Pillow installed
✓ OpenCV installed
✓ H5py installed
✓ Pandas installed
✓ Matplotlib installed
✓ scikit-learn installed
✓ tqdm installed

============================================================
  OpenSlide Check (Optional)
============================================================
✓ OpenSlide installed (version: 1.x.x)

============================================================
  GPU Check
============================================================
✓ CUDA available
  GPU 0: NVIDIA GeForce RTX 4070
  Total GPUs: 1
  VRAM: 8.00 GB

============================================================
  Project Structure Check
============================================================
✓ src/utils/
✓ src/preprocessing/
✓ src/feature_extraction/
✓ src/sampling/
✓ src/mil/
✓ src/explainability/
✓ configs/

============================================================
  Configuration File Check
============================================================
✓ config.yaml found
  ✓ Section 'experiment' present
  ✓ Section 'preprocessing' present
  ✓ Section 'feature_extraction' present
  ✓ Section 'sampling' present
  ✓ Section 'mil' present
  ✓ Section 'hardware' present
  ✓ Section 'paths' present
  ✓ Seed configured: 42
  ✓ Deterministic mode: True

============================================================
  Utility Modules Check
============================================================
✓ logger module OK
✓ seed module OK
✓ gpu_monitor module OK
✓ config module OK

============================================================
  Verification Summary
============================================================
✓ PASS   - Python
✓ PASS   - Dependencies
✓ PASS   - OpenSlide
✓ PASS   - GPU
✓ PASS   - Structure
✓ PASS   - Config
✓ PASS   - Utils

============================================================
✓ VERIFICATION SUCCESSFUL - Ready for Phase 0.2
============================================================
```

---

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution**: Install PyTorch
```bash
pip install torch torchvision
```

### Issue: "CUDA not available"
**Solution**: Install CUDA-enabled PyTorch
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "OpenSlide not found"
**Solution**: Install system-level OpenSlide library (see step 3 above)

### Issue: "Import errors for utils modules"
**Solution**: Install package in development mode
```bash
pip install -e .
```

---

## Minimal Installation (Without GPU)

If you don't have a GPU or just want to test the structure:

```bash
# Install core dependencies only
pip install numpy pandas matplotlib pyyaml tqdm scikit-learn

# Install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Note: Training will be VERY slow on CPU
```

---

## Next Steps After Verification

Once verification passes, you're ready for **Phase 0.2**:
- Implement WSI preprocessing (tissue detection + tiling)

Request to proceed:
```
"Proceed with Phase 0.2"
```
