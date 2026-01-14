"""
Test Utilities for User-Upload Inference Sandbox

Provides isolated test infrastructure utilities that ensure
complete separation from training/evaluation artifacts.

Safety guarantees:
- Never writes to data/, results/, or checkpoints/
- Uses test_data/ directory exclusively
- Validates input types before processing
- Maintains comprehensive audit logs
"""
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from datetime import datetime
import pandas as pd


def setup_test_directories(base_dir: Path) -> Dict[str, Path]:
    """
    Create isolated test directory structure.
    
    Args:
        base_dir: Root test directory (should be test_data/)
    
    Returns:
        Dictionary mapping directory names to paths
    """
    structure = {
        'input_wsi': base_dir / 'input' / 'wsi',
        'input_patches': base_dir / 'input' / 'patches_direct',
        'patches': base_dir / 'patches',
        'features': base_dir / 'features',
        'features_topk': base_dir / 'features_topk',
        'results': base_dir / 'test_results',
        'predictions': base_dir / 'test_results' / 'predictions',
        'heatmaps_mil': base_dir / 'test_results' / 'heatmaps' / 'mil_attention',
        'heatmaps_gradcam': base_dir / 'test_results' / 'heatmaps' / 'gradcam',
        'heatmaps_prob': base_dir / 'test_results' / 'heatmaps' / 'probability_maps',
        'overlays': base_dir / 'test_results' / 'overlays',
        'calibration': base_dir / 'test_results' / 'calibration',
        'logs': base_dir / 'test_results' / 'logs',
    }
    
    for dir_path in structure.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return structure


def detect_input_type(file_path: Path) -> Literal['wsi', 'patch', 'unknown']:
    """
    Detect input image type based on file extension.
    
    Args:
        file_path: Path to input image
    
    Returns:
        'wsi' for .tif/.tiff, 'patch' for .png/.jpg, 'unknown' otherwise
    """
    ext = file_path.suffix.lower()
    
    if ext in ['.tif', '.tiff']:
        return 'wsi'
    elif ext in ['.png', '.jpg', '.jpeg']:
        return 'patch'
    else:
        return 'unknown'


def save_test_prediction(
    prediction: Dict[str, Any],
    output_dir: Path,
    image_id: str
) -> Path:
    """
    Save prediction as JSON with timestamp and metadata.
    
    Args:
        prediction: Prediction dictionary
        output_dir: Output directory for predictions
        image_id: Unique image identifier
    
    Returns:
        Path to saved JSON file
    """
    # Add metadata
    prediction['timestamp'] = datetime.now().isoformat()
    prediction['image_id'] = image_id
    prediction['ground_truth'] = 'NOT_AVAILABLE'
    prediction['accuracy'] = 'NOT_COMPUTED'
    
    # Clinical disclaimer
    prediction['disclaimer'] = (
        "This is a model-based prediction for research purposes only. "
        "NOT a clinical diagnosis. NOT approved for medical use."
    )
    
    output_path = output_dir / f"{image_id}_prediction.json"
    with open(output_path, 'w') as f:
        json.dump(prediction, f, indent=2)
    
    return output_path


def append_to_confidence_csv(
    prediction: Dict[str, Any],
    csv_path: Path
) -> None:
    """
    Append prediction to aggregated confidence scores CSV.
    
    Args:
        prediction: Prediction dictionary
        csv_path: Path to confidence CSV file
    """
    row = {
        'image_id': prediction['image_id'],
        'timestamp': prediction['timestamp'],
        'predicted_label': prediction['predicted_label'],
        'confidence': prediction['confidence'],
        'probability': prediction.get('probability', prediction['confidence']),
        'model_checkpoint': prediction.get('model_checkpoint', 'unknown'),
    }
    
    df = pd.DataFrame([row])
    
    if csv_path.exists():
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)


def validate_test_isolation(output_dir: Path) -> bool:
    """
    Validate that test output directory is isolated from core artifacts.
    
    Args:
        output_dir: Proposed output directory
    
    Returns:
        True if isolated, False if risks contamination
    
    Raises:
        ValueError: If output directory overlaps with protected paths
    """
    protected_dirs = ['data', 'results', 'checkpoints', 'logs']
    
    output_str = str(output_dir.resolve()).lower()
    
    for protected in protected_dirs:
        if f"\\{protected}\\" in output_str or f"/{protected}/" in output_str:
            if 'test_data' not in output_str:
                raise ValueError(
                    f"Output directory overlaps with protected path: {protected}\n"
                    f"Test outputs must be under test_data/ only."
                )
    
    return True


def generate_test_summary_report(results_dir: Path) -> Path:
    """
    Generate summary report from test results.
    
    Args:
        results_dir: Test results directory
    
    Returns:
        Path to generated summary report
    """
    # Read confidence CSV
    csv_path = results_dir / 'confidence_scores.csv'
    
    if not csv_path.exists():
        summary = "# Test Summary Report\n\nNo predictions generated yet.\n"
    else:
        df = pd.read_csv(csv_path)
        
        summary = f"""# Test Summary Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

- **Total Predictions**: {len(df)}
- **Tumor Predictions**: {(df['predicted_label'] == 1).sum()}
- **Normal Predictions**: {(df['predicted_label'] == 0).sum()}

## Confidence Statistics

- **Mean Confidence**: {df['confidence'].mean():.2%}
- **Median Confidence**: {df['confidence'].median():.2%}
- **Min Confidence**: {df['confidence'].min():.2%}
- **Max Confidence**: {df['confidence'].max():.2%}

## Important Disclaimers

> [!WARNING]
> **Research Use Only**
> 
> These predictions are model-based inferences for research purposes.
> They are NOT clinical diagnoses and NOT approved for medical use.

> [!NOTE]
> **Ground Truth Not Available**
> 
> Accuracy metrics are NOT computed for user-uploaded test images.
> Confidence scores represent model certainty, not clinical accuracy.

## Output Locations

- Predictions: `test_results/predictions/`
- Heatmaps: `test_results/heatmaps/`
- Overlays: `test_results/overlays/`
- Logs: `test_results/logs/`

For detailed results, see individual prediction JSON files.
"""
    
    report_path = results_dir / 'summary_report.md'
    with open(report_path, 'w') as f:
        f.write(summary)
    
    return report_path


def create_test_readme(test_data_dir: Path) -> Path:
    """
    Create README for test_data directory.
    
    Args:
        test_data_dir: Test data root directory
    
    Returns:
        Path to created README
    """
    readme_content = """# Test Data Directory

## Purpose

This directory contains **user-uploaded test images** and their **inference outputs**.

It is **completely isolated** from training, evaluation, and production artifacts.

---

## Directory Structure

```
test_data/
├── input/
│   ├── wsi/                  # Place .tif/.tiff WSI images here
│   └── patches_direct/       # Place .png/.jpg patches here
│
├── patches/                  # Auto-generated patches (from WSI)
├── features/                 # Extracted features
├── features_topk/            # Top-K sampled features (if applicable)
│
└── test_results/
    ├── predictions/          # Per-image JSON predictions
    ├── heatmaps/             # MIL attention & Grad-CAM
    ├── overlays/             # Visualization overlays
    ├── calibration/          # Confidence analysis
    ├── logs/                 # Execution logs
    ├── confidence_scores.csv # Aggregated predictions
    └── summary_report.md     # Auto-generated summary
```

---

## Usage

### 1. Place Test Images

**For WSI (.tif/.tiff)**:
```bash
copy your_image.tif test_data/input/wsi/
```

**For Patches (.png/.jpg)**:
```bash
copy your_patch.png test_data/input/patches_direct/
```

### 2. Run Test Inference

```bash
python scripts/test_inference.py \\
    --input test_data/input/wsi/your_image.tif \\
    --model checkpoints/best_model.pth \\
    --output test_data/test_results
```

### 3. View Results

- **Prediction**: `test_data/test_results/predictions/<image_id>_prediction.json`
- **Summary**: `test_data/test_results/summary_report.md`
- **Heatmaps**: `test_data/test_results/heatmaps/`

---

## Important Notes

### ⚠️ Research Use Only

All outputs are **model-based predictions** for research purposes.

**NOT clinical diagnoses. NOT approved for medical use.**

### Ground Truth & Accuracy

- Ground truth labels are **NOT required** for test images
- Accuracy metrics are **NOT computed** (no ground truth available)
- Confidence scores represent **model certainty**, not clinical accuracy

### Isolation Guarantee

This directory **never writes** to:
- `data/` (training data)
- `results/` (evaluation results)
- `checkpoints/` (model weights)

---

## Troubleshooting

**No predictions generated?**
- Check `test_data/test_results/logs/inference.log`

**Missing heatmaps?**
- Some visualizations may be skipped if unsupported by model architecture

**Want to clear test results?**
```bash
rmdir /s test_data\\test_results
mkdir test_data\\test_results
```

---

For more information, see main `README.md`.
"""
    
    readme_path = test_data_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    return readme_path
