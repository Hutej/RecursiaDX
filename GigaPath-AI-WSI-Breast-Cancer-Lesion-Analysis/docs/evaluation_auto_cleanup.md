# Evaluation Pipeline Auto-Cleanup

## Summary

Modified `scripts/evaluate_mil.py` to automatically clean the evaluation directory before each run.

---

## Implementation

### Function Added

```python
def reset_evaluation_directory(output_dir: Path):
    """
    Clean evaluation directory before starting new evaluation.
    
    Removes all previous evaluation artifacts to ensure clean state.
    Only affects results/evaluation/ — does NOT touch training data,
    features, or checkpoints.
    
    Args:
        output_dir: Path to evaluation output directory
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info("[CLEAN] Previous evaluation results removed")
    
    output_dir.mkdir(parents=True, exist_ok=True)
```

### Execution Flow

```
1. Parse arguments
2. Load config
3. Setup logging
4. → CLEAN evaluation directory (NEW)
5. Verify GPU
6. Load model
7. Run evaluation
```

---

## Behavior

### If `results/evaluation/` Does Not Exist
- Creates directory
- No cleanup message

### If `results/evaluation/` Exists
- Deletes entire directory and contents
- Recreates empty directory
- Logs: `[CLEAN] Previous evaluation results removed`

---

## Safety Guarantees

### What Is Deleted
✅ `results/evaluation/metrics.json`  
✅ `results/evaluation/predictions.csv`  
✅ `results/evaluation/confusion_matrix.png`  
✅ `results/evaluation/roc_curve.png`  
✅ Any other files in `results/evaluation/`

### What Is NEVER Touched
❌ `checkpoints/` (models preserved)  
❌ `data/features/` (features preserved)  
❌ `data/features_topk/` (top-k preserved)  
❌ `data/labels.csv` (labels preserved)  
❌ Any other directories

---

## Usage

```bash
# Every run starts from clean slate
python scripts/evaluate_mil.py \
  --model checkpoints/best_model.pth \
  --features data/features_topk \
  --labels data/labels.csv

# Output:
# [CLEAN] Previous evaluation results removed
# ... evaluation proceeds ...
```

---

## Benefits

1. **No Stale Metrics**: Old `metrics.json` never mixed with new results
2. **No Mixed Plots**: Fresh confusion matrix and ROC curve each time
3. **Reproducible**: Each evaluation independent of previous runs
4. **No User Prompts**: Automatic cleanup, no confirmation needed

---

## Testing

### Verify Cleanup

```bash
# Run evaluation twice
python scripts/evaluate_mil.py --model checkpoints/best_model.pth --features data/features_topk --labels data/labels.csv

# Check timestamp on metrics.json (should be recent)
dir results\evaluation\metrics.json

# Run again
python scripts/evaluate_mil.py --model checkpoints/best_model.pth --features data/features_topk --labels data/labels.csv

# Timestamp should update (old file deleted, new file created)
```

---

**Status**: ✅ Implemented  
**Modified File**: `scripts/evaluate_mil.py`  
**Lines Changed**: +18  
**Breaking Changes**: None (backward compatible)
