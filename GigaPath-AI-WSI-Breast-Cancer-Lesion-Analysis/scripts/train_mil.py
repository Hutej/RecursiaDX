"""
MIL Training CLI Script

Train attention-based MIL model for slide-level classification.

Usage:
    # Train from scratch
    python scripts/train_mil.py --features data/features_topk --labels data/labels.csv
    
    # Resume from checkpoint
    python scripts/train_mil.py --features data/features_topk --labels data/labels.csv --resume checkpoints/last_model.pth
    
    # Custom config
    python scripts/train_mil.py --config configs/config.yaml
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mil import AttentionMIL, MILTrainer, MILDataset
from src.utils import setup_logger, load_config, set_seed, get_device
from src.utils.logger import get_logger

logger = None  # Will be initialized in main()


def create_data_splits(labels_csv: str, val_split: float = 0.2, test_split: float = 0.1, seed: int = 42):
    """
    Create train/val/test splits with stratification.
    
    Args:
        labels_csv: Path to labels CSV
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed
    
    Returns:
        Tuple of (train_csv, val_csv, test_csv) paths
    """
    df = pd.read_csv(labels_csv)
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_split, random_state=seed, stratify=df['label']
    )
    
    # Second split: train vs val
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_split/(1-test_split), random_state=seed, stratify=train_val_df['label']
    )
    
    # Save splits
    output_dir = Path(labels_csv).parent
    train_csv = output_dir / 'labels_train.csv'
    val_csv = output_dir / 'labels_val.csv'
    test_csv = output_dir / 'labels_test.csv'
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    logger.info(f"Data splits created:")
    logger.info(f"  Train: {len(train_df)} slides")
    logger.info(f"  Val: {len(val_df)} slides")
    logger.info(f"  Test: {len(test_df)} slides")
    
    return str(train_csv), str(val_csv), str(test_csv)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train Attention MIL model for slide-level classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
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
        help='Path to labels CSV (columns: slide_name, label)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume training'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Setup logging
    global logger
    logger = setup_logger(
        name='mil_training',
        log_dir=config['paths']['logs'],
        level=config['logging']['level'],
        save_to_file=config['logging']['save_logs']
    )
    
    # Set random seed
    set_seed(
        seed=config['experiment']['seed'],
        deterministic=config['experiment']['deterministic']
    )
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create data splits
    train_csv, val_csv, _ = create_data_splits(
        labels_csv=args.labels,
        seed=config['experiment']['seed']
    )
    
    # Create datasets
    train_dataset = MILDataset(hdf5_dir=args.features, labels_csv=train_csv)
    val_dataset = MILDataset(hdf5_dir=args.features, labels_csv=val_csv)
    
    # Create data loaders (batch_size=1 for MIL)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=MILDataset.collate_mil
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=MILDataset.collate_mil
    )
    
    logger.info(f"Datasets created:")
    logger.info(f"  Train: {len(train_dataset)} slides")
    logger.info(f"  Val: {len(val_dataset)} slides")
    
    # Create model
    model = AttentionMIL(
        input_dim=config['mil']['input_dim'],
        hidden_dim=config['mil']['hidden_dim'],
        attn_dim=config['mil']['attn_dim'],
        num_classes=config['mil']['num_classes'],
        dropout=config['mil']['dropout']
    )
    
    logger.info(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    if config['mil']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['mil']['learning_rate'],
            weight_decay=config['mil']['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['mil']['learning_rate'],
            weight_decay=config['mil']['weight_decay']
        )
    
    # Create loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Create learning rate scheduler
    if config['mil']['scheduler'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=config['mil']['scheduler_patience'],
            factor=config['mil']['scheduler_factor']
        )
    else:
        scheduler = None
    
    # Create trainer
    trainer = MILTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=config['paths']['checkpoints']
    )
    
    # Resume if requested
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Train
    num_epochs = args.epochs if args.epochs else config['mil']['num_epochs']
    
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Learning rate: {config['mil']['learning_rate']}")
    logger.info(f"Early stopping patience: {config['mil']['early_stopping_patience']}")
    logger.info("="*60)
    
    try:
        trainer.train(
            num_epochs=num_epochs,
            early_stopping_patience=config['mil']['early_stopping_patience'],
            scheduler=scheduler
        )
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Best validation AUC: {trainer.best_auc:.4f}")
        logger.info(f"Checkpoints saved to: {config['paths']['checkpoints']}")
        logger.info("="*60)
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
