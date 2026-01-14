"""
MIL Model Evaluation Pipeline

Comprehensive evaluation script for trained Attention MIL model.
Computes metrics, generates visualizations, and saves results.

Usage:
    python scripts/evaluate_mil.py \
        --model checkpoints/best_model.pth \
        --features data/features_topk \
        --labels data/labels.csv \
        --output results/evaluation

Requirements:
    - GPU-only execution (aborts if CUDA unavailable)
    - Uses TEST split from labels_test.csv
    - No retraining, evaluation only
"""
import argparse
import sys
import shutil
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    roc_curve
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mil import AttentionMIL, MILDataset
from src.utils import setup_logger, load_config
from src.utils.logger import get_logger

logger = None  # Will be initialized in main()


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


def enforce_gpu():
    """
    Enforce GPU-only execution. Abort if CUDA not available.
    
    Raises:
        RuntimeError: If CUDA is not available
    """
    if not torch.cuda.is_available():
        print("\n" + "="*60)
        print("❌ GPU ENFORCEMENT FAILED")
        print("="*60)
        print("CUDA is not available on this system.")
        print("This evaluation script requires GPU acceleration.")
        print("CPU fallback is DISABLED as per requirements.")
        print("="*60)
        raise RuntimeError("CUDA not available - GPU-only execution enforced")
    
    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(0)
    
    print("\n" + "="*60)
    print("✓ GPU VERIFICATION PASSED")
    print("="*60)
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print("CPU FALLBACK: DISABLED")
    print("="*60 + "\n")
    
    return device


def load_model(checkpoint_path: str, device: torch.device) -> AttentionMIL:
    """
    Load trained MIL model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded AttentionMIL model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with same architecture as training
    model = AttentionMIL(
        input_dim=2048,
        hidden_dim=512,
        attn_dim=256,
        num_classes=2,
        dropout=0.25
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"✓ Model loaded from {checkpoint_path}")
    logger.info(f"  Trained epoch: {checkpoint['epoch']}")
    logger.info(f"  Best validation AUC: {checkpoint.get('best_auc', 'N/A'):.4f}")
    
    return model


def evaluate_model(
    model: AttentionMIL,
    test_loader: DataLoader,
    device: torch.device
):
    """
    Run evaluation on test set.
    
    Args:
        model: Trained MIL model
        test_loader: Test data loader
        device: Device
    
    Returns:
        Dictionary with predictions, labels, probabilities, logits, and slide names
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_logits = []
    all_slide_names = []
    
    logger.info("Running inference on test set...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Get features and label
            features = batch['features'].to(device)
            label = batch['label'].item()
            slide_name = batch['slide_name']
            
            # Forward pass
            logit, _ = model.forward(features, return_attention=False)
            
            # Get probability and prediction
            prob = torch.sigmoid(logit).item()
            prediction = 1 if prob > 0.5 else 0
            
            # Store results
            all_predictions.append(prediction)
            all_labels.append(label)
            all_probabilities.append(prob)
            all_logits.append(logit.item())
            all_slide_names.append(slide_name)
    
    return {
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probabilities),
        'logits': np.array(all_logits),
        'slide_names': all_slide_names
    }


def compute_metrics(labels, predictions, probabilities):
    """
    Compute evaluation metrics.
    
    Args:
        labels: Ground truth labels
        predictions: Predicted labels
        probabilities: Predicted probabilities
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': float(accuracy_score(labels, predictions)),
        'roc_auc': float(roc_auc_score(labels, probabilities)),
        'precision': float(precision_score(labels, predictions, zero_division=0)),
        'recall': float(recall_score(labels, predictions, zero_division=0)),
        'f1_score': float(f1_score(labels, predictions, zero_division=0)),
        'num_samples': int(len(labels)),
        'num_positive': int(np.sum(labels)),
        'num_negative': int(len(labels) - np.sum(labels))
    }
    
    return metrics


def plot_confusion_matrix(labels, predictions, output_path):
    """
    Generate and save confusion matrix visualization.
    
    Args:
        labels: Ground truth labels
        predictions: Predicted labels
        output_path: Path to save plot
    """
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Benign (0)', 'Malignant (1)'],
        yticklabels=['Benign (0)', 'Malignant (1)'],
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Test Set Evaluation', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Confusion matrix saved to {output_path}")


def plot_roc_curve(labels, probabilities, output_path, auc_score):
    """
    Generate and save ROC curve visualization.
    
    Args:
        labels: Ground truth labels
        probabilities: Predicted probabilities
        output_path: Path to save plot
        auc_score: AUC score for display
    """
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Test Set Evaluation', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ ROC curve saved to {output_path}")


def save_results(results, metrics, output_dir):
    """
    Save evaluation results to files.
    
    Args:
        results: Dictionary with predictions and labels
        metrics: Dictionary of computed metrics
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics to JSON
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✓ Metrics saved to {metrics_path}")
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'slide_name': results['slide_names'],
        'true_label': results['labels'],
        'predicted_label': results['predictions'],
        'probability': results['probabilities'],
        'logit': results['logits']
    })
    predictions_path = output_dir / 'predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"✓ Predictions saved to {predictions_path}")
    
    # Generate confusion matrix
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(results['labels'], results['predictions'], cm_path)
    
    # Generate ROC curve
    roc_path = output_dir / 'roc_curve.png'
    plot_roc_curve(results['labels'], results['probabilities'], roc_path, metrics['roc_auc'])


def print_metrics_summary(metrics):
    """
    Print formatted metrics summary.
    
    Args:
        metrics: Dictionary of computed metrics
    """
    logger.info("\n" + "="*60)
    logger.info("EVALUATION METRICS - TEST SET")
    logger.info("="*60)
    logger.info(f"Total Samples:       {metrics['num_samples']}")
    logger.info(f"  Benign (0):        {metrics['num_negative']}")
    logger.info(f"  Malignant (1):     {metrics['num_positive']}")
    logger.info("-"*60)
    logger.info(f"Accuracy:            {metrics['accuracy']:.4f}")
    logger.info(f"ROC AUC:             {metrics['roc_auc']:.4f}")
    logger.info(f"Precision:           {metrics['precision']:.4f}")
    logger.info(f"Recall:              {metrics['recall']:.4f}")
    logger.info(f"F1-Score:            {metrics['f1_score']:.4f}")
    logger.info("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained Attention MIL model on test set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='Directory containing HDF5 feature files'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='Path to labels CSV (will use labels_test.csv for test split)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/evaluation',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # ENSURE OUTPUT DIRECTORY EXISTS (PREVENT SILENT FAILURES)
    # ========================================================================
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: GPU ENFORCEMENT (CRITICAL - NO CPU FALLBACK)
    # ========================================================================
    try:
        device = enforce_gpu()
    except RuntimeError as e:
        sys.exit(1)
    
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Setup logging
    global logger
    logger = setup_logger(
        name='mil_evaluation',
        log_dir=config['paths']['logs'],
        level=config['logging']['level'],
        save_to_file=config['logging']['save_logs']
    )
    
    logger.info("="*60)
    logger.info("MIL MODEL EVALUATION PIPELINE")
    logger.info("="*60)
    logger.info(f"Model checkpoint:    {args.model}")
    logger.info(f"Features directory:  {args.features}")
    logger.info(f"Labels file:         {args.labels}")
    logger.info(f"Output directory:    {args.output}")
    logger.info("="*60)
    
    # CLEAN: Reset evaluation directory
    output_dir = Path(args.output)
    reset_evaluation_directory(output_dir)
    logger.info(f"Evaluation directory ready: {output_dir.absolute()}")
    logger.info("="*60)
    
    try:
        # ====================================================================
        # STEP 2: LOAD MODEL
        # ====================================================================
        model = load_model(args.model, device)
        
        # ====================================================================
        # STEP 3: LOAD TEST DATA
        # ====================================================================
        # Use test split (same logic as train_mil.py)
        labels_path = Path(args.labels)
        test_csv = labels_path.parent / 'labels_test.csv'
        
        if not test_csv.exists():
            logger.error(f"Test split CSV not found: {test_csv}")
            logger.error("Please run train_mil.py first to create data splits.")
            sys.exit(1)
        
        logger.info(f"Loading test split from {test_csv}")
        test_dataset = MILDataset(hdf5_dir=args.features, labels_csv=str(test_csv))
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=MILDataset.collate_mil
        )
        
        logger.info(f"Test dataset: {len(test_dataset)} slides")
        
        # ====================================================================
        # STEP 4: RUN EVALUATION (NO GRADIENTS)
        # ====================================================================
        results = evaluate_model(model, test_loader, device)
        
        # ====================================================================
        # STEP 5: COMPUTE METRICS
        # ====================================================================
        metrics = compute_metrics(
            results['labels'],
            results['predictions'],
            results['probabilities']
        )
        
        # ====================================================================
        # STEP 6: SAVE RESULTS AND VISUALIZATIONS
        # ====================================================================
        save_results(results, metrics, args.output)
        
        # ====================================================================
        # STEP 7: PRINT SUMMARY
        # ====================================================================
        print_metrics_summary(metrics)
        
        logger.info(f"✓ All results saved to {args.output}/")
        logger.info("\n✅ EVALUATION COMPLETE!\n")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
