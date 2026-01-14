"""
Tile ranking module for Top-K sampling.
Ranks tiles by informativeness for efficient MIL training.
"""
import numpy as np
import torch
from typing import Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TileRanker:
    """Rank tiles by informativeness."""
    
    @staticmethod
    def normalize_features(features: np.ndarray) -> np.ndarray:
        """
        L2-normalize feature vectors to unit norm.
        
        CRITICAL for reproducibility: Prevents scale bias in ranking.
        Each feature vector is normalized independently: f_norm = f / ||f||_2
        
        Args:
            features: Feature array (N, feature_dim)
        
        Returns:
            L2-normalized features (N, feature_dim) with unit norm
        """
        # Compute L2 norms
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        
        # Prevent division by zero (should never happen with real features)
        norms = np.maximum(norms, 1e-12)
        
        # Normalize
        normalized_features = features / norms
        
        # Verify unit norms (debug)
        verify_norms = np.linalg.norm(normalized_features, axis=1)
        logger.debug(f"Feature normalization: mean_norm={verify_norms.mean():.6f}, std_norm={verify_norms.std():.6f}")
        
        # Log statistics
        logger.info(f"Normalized {features.shape[0]} features to unit L2 norm")
        logger.debug(f"  Original norm range: [{norms.min():.2f}, {norms.max():.2f}]")
        logger.debug(f"  Normalized norm check: [{verify_norms.min():.6f}, {verify_norms.max():.6f}]")
        
        return normalized_features
    
    @staticmethod
    def rank_by_feature_norm(features: np.ndarray, normalize_first: bool = True) -> np.ndarray:
        """
        Rank tiles by L2 norm of feature vectors.
        
        Higher norm indicates stronger activations = more informative.
        
        Args:
            features: Feature array (N, feature_dim)
            normalize_first: If True, normalize features before ranking (recommended for reproducibility)
        
        Returns:
            Ranking scores (N,) - higher is better
        """
        # Apply normalization if requested
        if normalize_first:
            features = TileRanker.normalize_features(features)
            logger.debug("Features normalized before ranking (reproducible mode)")
        
        scores = np.linalg.norm(features, axis=1)
        logger.debug(f"Feature norm ranking: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")
        return scores
    
    @staticmethod
    def normalize_scores(scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        Args:
            scores: Raw scores (N,)
        
        Returns:
            Normalized scores (N,) in [0, 1] range
        """
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # Handle edge case: all scores are identical
        if max_score - min_score < 1e-10:
            logger.warning("All scores are identical, returning uniform normalized scores")
            return np.ones_like(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        logger.debug(f"Normalized scores: min={normalized.min():.4f}, max={normalized.max():.4f}")
        return normalized
    
    @staticmethod
    def rank_by_attention(
        features: np.ndarray,
        mil_model: torch.nn.Module,
        device: Optional[torch.device] = None
    ) -> np.ndarray:
        """
        Rank tiles by attention weights from trained MIL model.
        
        Note: Requires trained MIL model (Phase 3).
        
        Args:
            features: Feature array (N, feature_dim)
            mil_model: Trained MIL model with attention mechanism
            device: Torch device
        
        Returns:
            Attention scores (N,) - higher is better
        """
        if device is None:
            device = torch.device('cuda')
        
        if device.type != 'cuda':
             raise ValueError(f"Device must be CUDA, got {device.type}")

        # Convert to tensor
        features_tensor = torch.from_numpy(features).float().to(device)
        
        # Get attention weights
        mil_model.eval()
        with torch.no_grad():
            # Assuming model has get_attention_weights method
            attention = mil_model.get_attention_weights(features_tensor)
        
        # Convert to numpy
        scores = attention.cpu().numpy()
        
        logger.debug(f"Attention ranking: min={scores.min():.4f}, max={scores.max():.4f}")
        return scores
    
    @staticmethod
    def rank_by_weighted_combination(
        features: np.ndarray,
        mil_model: torch.nn.Module,
        alpha: float = 0.7,
        device: Optional[torch.device] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rank tiles by weighted combination of attention and L2-norm scores.
        
        Best-practice approach (Strategy-1):
        final_score = α * attention_score + (1 - α) * l2_norm_score
        
        Both scores are normalized to [0, 1] before combining.
        
        Args:
            features: Feature array (N, feature_dim)
            mil_model: Trained MIL model with attention mechanism
            alpha: Weight for attention scores (default: 0.7)
            device: Torch device
        
        Returns:
            Tuple of (final_scores, attention_scores, l2_norm_scores)
            - final_scores: Weighted combination (N,)
            - attention_scores: Normalized attention weights (N,)
            - l2_norm_scores: Normalized L2 norms (N,)
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        
        if device is None:
            device = torch.device('cuda')
        
        logger.info(f"Computing weighted scores with α={alpha:.2f}")
        
        # Compute attention scores
        attention_scores_raw = TileRanker.rank_by_attention(features, mil_model, device)
        attention_scores = TileRanker.normalize_scores(attention_scores_raw)
        logger.debug(f"Attention scores normalized: shape={attention_scores.shape}")
        
        # Compute L2-norm scores
        l2_norm_scores_raw = TileRanker.rank_by_feature_norm(features)
        l2_norm_scores = TileRanker.normalize_scores(l2_norm_scores_raw)
        logger.debug(f"L2-norm scores normalized: shape={l2_norm_scores.shape}")
        
        # Weighted combination
        final_scores = alpha * attention_scores + (1 - alpha) * l2_norm_scores
        
        logger.info(
            f"Final scores: min={final_scores.min():.4f}, "
            f"max={final_scores.max():.4f}, mean={final_scores.mean():.4f}"
        )
        logger.info(
            f"Score composition: {alpha:.0%} attention + {(1-alpha):.0%} L2-norm"
        )
        
        return final_scores, attention_scores, l2_norm_scores
    
    @staticmethod
    def select_top_k(scores: np.ndarray, k: int) -> np.ndarray:
        """
        Select indices of top-K scores.
        
        Args:
            scores: Ranking scores (N,)
            k: Number of top tiles to select
        
        Returns:
            Indices of top-K tiles (K,) in descending score order
        """
        n = len(scores)
        
        # Handle case where k >= n
        if k >= n:
            logger.warning(f"K={k} >= num_tiles={n}, keeping all tiles")
            # Return all indices sorted by score (descending)
            return np.argsort(scores)[::-1]
        
        # Get top-k indices (descending order)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        logger.debug(f"Selected top {k} tiles from {n} (ratio: {k/n:.2%})")
        
        return top_k_indices
    
    @staticmethod
    def get_score_stats(scores: np.ndarray) -> dict:
        """
        Compute summary statistics of scores.
        
        Storage optimization: Store only stats, not full array.
        
        Args:
            scores: Ranking scores (N,)
        
        Returns:
            Dictionary with min, max, mean, std
        """
        stats = {
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores))
        }
        return stats


if __name__ == '__main__':
    # Test tile ranker
    print("Testing TileRanker...")
    
    # Create dummy features
    features = np.random.randn(2000, 2048).astype(np.float32)
    
    # Test feature norm ranking
    scores = TileRanker.rank_by_feature_norm(features)
    print(f"✓ Feature norm scores: shape={scores.shape}")
    
    # Test top-K selection
    k = 1000
    top_k_indices = TileRanker.select_top_k(scores, k)
    print(f"✓ Top-K selection: selected {len(top_k_indices)} / {len(scores)}")
    
    # Verify descending order
    assert all(scores[top_k_indices[i]] >= scores[top_k_indices[i+1]] 
               for i in range(len(top_k_indices)-1)), "Scores not in descending order!"
    print(f"✓ Scores in descending order")
    
    # Test score stats
    stats = TileRanker.get_score_stats(scores[top_k_indices])
    print(f"✓ Score stats: {stats}")
    
    print("\n✅ TileRanker tests passed!")
