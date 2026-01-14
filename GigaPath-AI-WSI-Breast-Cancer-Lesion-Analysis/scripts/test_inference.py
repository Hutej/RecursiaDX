"""
User-Upload Test Inference Pipeline

INFERENCE-ONLY sandbox for user-uploaded WSI and patch images.Reuses existing preprocessing, feature extraction, and MIL inference logic
but routes all outputs to isolated test_data/ directory.

SAFETY GUARANTEES:
- Never modifies training/evaluation artifacts
- Never writes to data/ or results/
- Uses trained model in eval mode only
- No gradient computation
- Complete audit logging

Usage:
    # WSI input
    python scripts/test_inference.py \\
        --input test_data/input/wsi/sample.tif \\
        --model checkpoints/best_model.pth
    
    # Patch input
    python scripts/test_inference.py \\
        --input test_data/input/patches_direct/patch.png \\
        --model checkpoints/best_model.pth
"""
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import torch
import h5py
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mil import AttentionMIL
from src.utils import load_config
from src.utils.test_utils import (
    setup_test_directories,
    detect_input_type,
    save_test_prediction,
    append_to_confidence_csv,
    validate_test_isolation,
    generate_test_summary_report,
    create_test_readme
)


def setup_test_logger(log_dir: Path) -> logging.Logger:
    """Set up test-specific logger."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"test_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger('test_inference')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_test_model(checkpoint_path: Path, device: torch.device, logger: logging.Logger) -> AttentionMIL:
    """Load trained MIL model in EVAL mode only."""
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = AttentionMIL(
        input_dim=2048,
        hidden_dim=512,
        attn_dim=256,
        num_classes=2,
        dropout=0.25
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # CRITICAL: Eval mode only, no training
    
    logger.info(f"Model loaded: epoch {checkpoint['epoch']}, AUC {checkpoint.get('best_auc', 'N/A')}")
    logger.info("Model set to EVALUATION MODE (no gradients)")
    
    return model


def process_wsi_input(
    wsi_path: Path,
    test_dirs: dict,
    logger: logging.Logger
) -> Path:
    """
    Process WSI input: tiling → feature extraction.
    
    Reuses existing preprocessing logic but writes to test_data/.
    
    Returns:
        Path to extracted features HDF5 file
    """
    logger.info(f"Processing WSI: {wsi_path.name}")
    
    # Import preprocessing utilities
    from src.preprocessing import TileExtractor
    from src.feature_extraction import FeatureExtractor
    
    slide_name = wsi_path.stem
    
    # Step 1: Extract tiles (writes to test_data/patches/)
    logger.info("Step 1/2: Extracting tiles...")
    tile_extractor = TileExtractor(
        tile_size=256,
        overlap=0,
        tissue_threshold=0.5
    )
    
    tiles_dir = test_dirs['patches'] / slide_name
    tiles_dir.mkdir(exist_ok=True)
    
    try:
        tile_extractor.extract_tiles(
            wsi_path=str(wsi_path),
            output_dir=str(tiles_dir)
        )
        logger.info(f"Tiles extracted to {tiles_dir}")
    except Exception as e:
        logger.error(f"Tile extraction failed: {e}")
        raise
    
    # Step 2: Extract features (writes to test_data/features/)
    logger.info("Step 2/2: Extracting features...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = FeatureExtractor(
        model_name='resnet50',
        device=device,
        use_amp=True
    )
    
    features_output = test_dirs['features'] / f"{slide_name}.h5"
    
    try:
        feature_extractor.extract_features_from_slide(
            slide_path=str(wsi_path),
            tiles_dir=str(tiles_dir),
            output_path=str(features_output)
        )
        logger.info(f"Features extracted to {features_output}")
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise
    
    return features_output


def process_patch_input(
    patch_path: Path,
    test_dirs: dict,
    logger: logging.Logger
) -> Path:
    """
    Process patch input: feature extraction only (no tiling).
    
    Returns:
        Path to extracted features HDF5 file
    """
    logger.info(f"Processing patch: {patch_path.name}")
    
    from src.feature_extraction import FeatureExtractor
    
    patch_name = patch_path.stem
    
    # Extract features directly from patch
    logger.info("Extracting features from patch...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = FeatureExtractor(
        model_name='resnet50',
        device=device,
        use_amp=True
    )
    
    features_output = test_dirs['features'] / f"{patch_name}.h5"
    
    try:
        # For single patch, create minimal HDF5
        from PIL import Image
        import torchvision.transforms as transforms
        
        img = Image.open(patch_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            features = feature_extractor.model(img_tensor)
            features = features.cpu().numpy()
        
        # Save to HDF5
        with h5py.File(features_output, 'w') as f:
            f.create_dataset('features', data=features)
            f.create_dataset('coords', data=np.array([[0, 0]]))
        
        logger.info(f"Features extracted to {features_output}")
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise
    
    return features_output


def run_mil_inference(
    features_path: Path,
    model: AttentionMIL,
    device: torch.device,
    logger: logging.Logger
) -> dict:
    """
    Run MIL inference on extracted features.
    
    Returns:
        Prediction dictionary
    """
    logger.info(f"Running MIL inference on {features_path.name}")
    
    # Load features
    with h5py.File(features_path, 'r') as f:
        features = torch.from_numpy(f['features'][:]).float().to(device)
    
    # Run inference (no gradients)
    with torch.no_grad():
        prediction_dict = model.predict_slide(features, return_attention=True)
    
    # Build result
    result = {
        'predicted_label': int(prediction_dict['prediction']),
        'predicted_class': 'Tumor' if prediction_dict['prediction'] == 1 else 'Normal',
        'confidence': float(prediction_dict['probability']),
        'probability': float(prediction_dict['probability']),
        'logit': float(prediction_dict['logit']),
        'model_checkpoint': 'best_model.pth',
    }
    
    logger.info(f"Prediction: {result['predicted_class']} ({result['confidence']:.2%} confidence)")
    
    return result, prediction_dict.get('attention_weights')


def generate_test_heatmaps(
    features_path: Path,
    attention_weights: np.ndarray,
    output_dirs: dict,
    image_id: str,
    logger: logging.Logger
) -> None:
    """
    Generate MIL attention heatmaps (fail-safe).
    
    Skips if not possible without breaking existing logic.
    """
    logger.info("Generating attention heatmaps...")
    
    try:
        # Load coordinates
        with h5py.File(features_path, 'r') as f:
            coords = f['coords'][:]
        
        # Save raw attention
        attention_path = output_dirs['heatmaps_mil'] / f"{image_id}_attention.npy"
        np.save(attention_path, attention_weights)
        
        logger.info(f"Attention weights saved to {attention_path}")
        logger.info(f"Shape: {attention_weights.shape}, Min: {attention_weights.min():.4f}, Max: {attention_weights.max():.4f}")
        
        # TODO: Add visualization generation (optional, fail-safe)
        
    except Exception as e:
        logger.warning(f"Heatmap generation skipped: {e}")
        logger.warning("Inference completed successfully despite visualization failure")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='User-upload test inference (isolated sandbox)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input image (.tif/.tiff for WSI, .png/.jpg for patch)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--test-dir',
        type=str,
        default='test_data',
        help='Test data root directory'
    )
    
    args = parser.parse_args()
    
    # Setup test directories
    test_base = Path(args.test_dir)
    test_dirs = setup_test_directories(test_base)
    
    # Validate isolation
    validate_test_isolation(test_dirs['results'])
    
    # Setup logger
    logger = setup_test_logger(test_dirs['logs'])
    
    logger.info("="*60)
    logger.info("USER-UPLOAD TEST INFERENCE (ISOLATED SANDBOX)")
    logger.info("="*60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Test directory: {test_base}")
    logger.info("="*60)
    
    # Parse input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    input_type = detect_input_type(input_path)
    image_id = input_path.stem
    
    logger.info(f"Input type: {input_type.upper()}")
    logger.info(f"Image ID: {image_id}")
    
    # GPU check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    try:
        # Load model
        model = load_test_model(Path(args.model), device, logger)
        
        # Process input based on type
        if input_type == 'wsi':
            features_path = process_wsi_input(input_path, test_dirs, logger)
        elif input_type == 'patch':
            features_path = process_patch_input(input_path, test_dirs, logger)
        else:
            logger.error(f"Unsupported file type: {input_path.suffix}")
            sys.exit(1)
        
        # Run MIL inference
        prediction, attention_weights = run_mil_inference(features_path, model, device, logger)
        
        # Save prediction
        pred_path = save_test_prediction(prediction, test_dirs['predictions'], image_id)
        logger.info(f"Prediction saved to {pred_path}")
        
        # Append to confidence CSV
        csv_path = test_dirs['results'] / 'confidence_scores.csv'
        append_to_confidence_csv(prediction, csv_path)
        logger.info(f"Confidence appended to {csv_path}")
        
        # Generate heatmaps (fail-safe)
        if attention_weights is not None:
            generate_test_heatmaps(features_path, attention_weights, test_dirs, image_id, logger)
        
        # Generate summary report
        summary_path = generate_test_summary_report(test_dirs['results'])
        logger.info(f"Summary report generated: {summary_path}")
        
        # Create README if missing
        readme_path = test_base / 'README.md'
        if not readme_path.exists():
            create_test_readme(test_base)
        
        logger.info("="*60)
        logger.info("✅ TEST INFERENCE COMPLETE")
        logger.info("="*60)
        logger.info(f"Results: {test_dirs['results']}")
        logger.info("Ground truth: NOT_AVAILABLE")
        logger.info("Accuracy: NOT_COMPUTED")
        logger.info("="*60)
    
    except Exception as e:
        logger.error(f"Test inference failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
