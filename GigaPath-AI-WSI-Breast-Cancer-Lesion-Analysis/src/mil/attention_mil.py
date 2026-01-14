"""
Attention-based Multiple Instance Learning (MIL) model.
Gated attention mechanism for slide-level classification from tile features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class GatedAttention(nn.Module):
    """
    Gated attention mechanism for MIL.
    
    Combines attention branch (V) and gate branch (U) for stable learning.
    Reference: CLAM (Lu et al., 2021)
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.25
    ):
        """
        Initialize gated attention.
        
        Args:
            input_dim: Input feature dimension (2048 for ResNet50)
            hidden_dim: Hidden layer dimension
            attn_dim: Attention dimension
            dropout: Dropout probability
        """
        super(GatedAttention, self).__init__()
        
        # Attention branch V (what to focus on)
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, attn_dim)
        )
        
        # Gate branch U (how much to focus)
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, attn_dim),
            nn.Sigmoid()
        )
        
        # Attention weights projection
        self.attention_w = nn.Linear(attn_dim, 1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights for tile features.
        
        Args:
            features: Tile features (K, input_dim)
        
        Returns:
            Attention weights (K, 1) - normalized to sum to 1
        """
        # Attention branch: what to look at
        A_V = self.attention_V(features)  # (K, attn_dim)
        
        # Gate branch: how much to look
        A_U = self.attention_U(features)  # (K, attn_dim)
        
        # Gated attention: element-wise multiplication
        A = A_V * A_U  # (K, attn_dim)
        
        # Attention scores
        attention_scores = self.attention_w(A)  # (K, 1)
        
        # Normalize attention weights
        attention_weights = F.softmax(attention_scores, dim=0)
        # CRITICAL: Softmax is applied across the tile dimension (K) because each forward pass
        # processes a single WSI bag (batch size = 1). This normalizes attention weights
        # over all tiles in the slide such that sum(alpha_i) = 1.
        # DO NOT change dim=0 to dim=1 if attempting batch-wise MIL without restructuring
        # the entire pipeline to handle multiple bags simultaneously.
        
        return attention_weights


class AttentionMIL(nn.Module):
    """
    Attention-based MIL model for slide-level classification.
    
    Architecture:
        Features (K, 2048) → Gated Attention → Slide Embedding (2048) → Classifier → Logits
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.25
    ):
        """
        Initialize Attention MIL model.
        
        Args:
            input_dim: Input feature dimension (2048 for ResNet50)
            hidden_dim: Hidden layer dimension
            attn_dim: Attention dimension
            num_classes: Number of classes (2 for binary, ignored - always uses 1 logit)
            dropout: Dropout probability
        """
        super(AttentionMIL, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Gated attention mechanism
        self.attention = GatedAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attn_dim=attn_dim,
            dropout=dropout
        )
        
        # Slide-level binary classifier (single logit for BCEWithLogitsLoss)
        # CRITICAL: Binary MIL uses 1 logit, not 2 (CLAM-style)
        self.classifier = nn.Linear(input_dim, 1)
        
        logger.info(f"AttentionMIL initialized: input_dim={input_dim}, num_classes={num_classes}")
    
    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for MIL classification.
        
        Args:
            features: Tile features (K, input_dim) - must be float32 on correct device
            return_attention: If True, return attention weights
        
        Returns:
            Tuple of (logit, attention_weights)
            - logit: (1,) single logit for binary classification
            - attention_weights: (K,) if return_attention else None
        """
        # Explicit float32 enforcement to prevent silent float64 usage
        # CRITICAL: float64 doubles VRAM and slows computation
        features = features.float()
        
        # Get attention weights
        attention_weights = self.attention(features)  # (K, 1)
        
        # Aggregate features using attention
        slide_embedding = torch.sum(attention_weights * features, dim=0)  # (input_dim,)
        
        # Classify (single logit for binary)
        logit = self.classifier(slide_embedding)  # (1,)
        
        if return_attention:
            return logit, attention_weights.squeeze()  # (1,), (K,)
        else:
            return logit, None
    
    def get_attention_weights(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights without gradient.
        
        For explainability (Stage 4).
        
        Args:
            features: Tile features (K, input_dim)
        
        Returns:
            Attention weights (K,) as numpy array
        """
        self.eval()
        with torch.no_grad():
            attention_weights = self.attention(features)
        return attention_weights.squeeze().cpu().numpy()
    
    def predict_slide(
        self,
        features: torch.Tensor,
        return_attention: bool = True
    ) -> dict:
        """
        Predict slide-level label with attention weights.
        
        Args:
            features: Tile features (K, input_dim)
            return_attention: If True, return attention weights
        
        Returns:
            Dictionary with prediction, probability, and attention
        """
        self.eval()
        with torch.no_grad():
            logit, attention_weights = self.forward(features, return_attention=True)
            
            # Binary classification with single logit
            # Sigmoid gives probability of positive class
            prob = torch.sigmoid(logit).item()
            prediction = 1 if prob > 0.5 else 0
            
            # Legacy multi-class path (not used for binary MIL)
            if False:  # Disabled for binary MIL
                pass  # Multi-class not implemented
        
        result = {
            'prediction': prediction,
            'probability': prob,
            'logit': logit.item()
        }
        
        if return_attention:
            result['attention_weights'] = attention_weights.cpu().numpy()
        
        return result


if __name__ == '__main__':
    # Test AttentionMIL
    print("Testing AttentionMIL...")
    
    # Create model
    model = AttentionMIL(input_dim=2048, num_classes=2)
    
    # Create dummy features (K=1000 tiles, 2048-dim)
    features = torch.randn(1000, 2048)
    
    # Forward pass
    logits, attention_weights = model.forward(features, return_attention=True)
    
    print(f"✓ Input features: {features.shape}")
    print(f"✓ Output logits: {logits.shape}")
    print(f"✓ Attention weights: {attention_weights.shape}")
    
    # Check attention normalization
    attention_sum = attention_weights.sum().item()
    print(f"✓ Attention sum: {attention_sum:.6f} (should be ~1.0)")
    assert abs(attention_sum - 1.0) < 1e-4, "Attention weights don't sum to 1!"
    
    # Test prediction
    result = model.predict_slide(features)
    print(f"✓ Prediction: {result['prediction']}")
    print(f"✓ Probability: {result['probability']:.4f}")
    print(f"✓ Attention weights shape: {result['attention_weights'].shape}")
    
    print("\n✅ AttentionMIL tests passed!")
