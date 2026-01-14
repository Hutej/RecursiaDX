# Checkpoints Directory

This directory contains **trained model weights** that ARE committed to this repository.

## ✅ What's Included

The following trained checkpoints are **version-controlled** in this repository:

- `best_model.pth` (27 MB) - Best-performing model (highest validation AUC)
- `last_model.pth` (27 MB) - Latest training epoch checkpoint

## 🚀 Immediate Usage

After cloning this repository, you can run inference **immediately**:

```bash
python scripts/infer_mil.py \
    --model checkpoints/best_model.pth \
    --features data/features_topk \
    --output results/predictions.csv
```

## 🔄 Switching Checkpoints

To use the last epoch instead of the best model:

```bash
python scripts/infer_mil.py \
    --model checkpoints/last_model.pth \
    --features data/features_topk \
    --output results/predictions.csv
```

## ⚠️ Important Notes

1. **Checkpoints ARE committed**: These `.pth` files are tracked in Git for deployment simplicity.
2. **Datasets are NOT committed**: Training data must be obtained separately.
3. **No retraining needed**: Backend integration only requires these checkpoints.

---

**Repository Policy**: Trained model weights (< 30MB each) are committed for seamless backend deployment.
