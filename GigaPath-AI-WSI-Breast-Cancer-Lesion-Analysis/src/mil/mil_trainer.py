"""
MIL Trainer for slide-level classification.
Training loop with validation, metrics, and checkpointing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

from .attention_mil import AttentionMIL
from .mil_dataset import MILDataset
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MILTrainer:
    """Trainer for Attention MIL model."""
    
    def __init__(
        self,
        model: AttentionMIL,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Initialize MIL trainer.
        
        Args:
            model: AttentionMIL model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function (BCEWithLogitsLoss)
            optimizer: Optimizer
            device: Device (cuda/cpu)
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_auc = 0.0
        self.history = {'train_loss': [], 'val_metrics': []}
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(self.train_loader, desc='Training', leave=False):
            # Get data (batch_size=1 for MIL)
            features = batch['features'].to(self.device)  # (K, 2048)
            label = batch['label'].float().to(self.device)  # ()
            
            # Forward pass
            logit, _ = self.model(features, return_attention=False)  # (1,)
            
            # Compute loss
            loss = self.criterion(logit.squeeze(), label)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            Dictionary with metrics (auc, accuracy, f1)
        """
        self.model.eval()
        all_labels = []
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation', leave=False):
                features = batch['features'].to(self.device)
                label = batch['label'].item()
                
                # Forward pass
                logit, _ = self.model(features, return_attention=False)
                
                # Get probability
                prob = torch.sigmoid(logit).item()
                pred = 1 if prob > 0.5 else 0
                
                all_labels.append(label)
                all_probs.append(prob)
                all_preds.append(pred)
        
        # Compute metrics
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        
        metrics = {
            'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, zero_division=0)
        }
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: If True, save as best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_auc': self.best_auc
        }
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last_model.pth'
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"[OK] Saved best model (AUC: {metrics['auc']:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint for resuming training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_auc = checkpoint.get('best_auc', 0.0)
        
        logger.info(f"[OK] Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            scheduler: Learning rate scheduler (optional)
        """
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            logger.info("=" * 60)
            
            # Train
            train_loss = self.train_epoch()
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Val AUC: {val_metrics['auc']:.4f} | "
                       f"Acc: {val_metrics['accuracy']:.4f} | "
                       f"F1: {val_metrics['f1']:.4f}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Check for best model
            is_best = val_metrics['auc'] > self.best_auc
            if is_best:
                self.best_auc = val_metrics['auc']
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step(val_metrics['auc'])
                logger.info(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"\nEarly stopping triggered (no improvement for {early_stopping_patience} epochs)")
                break
        
        logger.info(f"\nTraining complete! Best AUC: {self.best_auc:.4f}")


if __name__ == '__main__':
    print("MILTrainer module - use train_mil.py for training")
