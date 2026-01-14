"""
Top-K tile selector for filtering HDF5 feature files.
Reduces computational cost by keeping only most informative tiles.
"""
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional

from .tile_ranker import TileRanker
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TopKSelector:
    """Select top-K tiles from HDF5 feature files."""
    
    def __init__(
        self, 
        k: int = 1000, 
        ranking_method: str = 'feature_norm',
        alpha: float = 0.7,
        mil_model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        normalize_features: bool = True
    ):
        """
        Initialize Top-K selector.
        
        Args:
            k: Number of top tiles to select
            ranking_method: 'feature_norm', 'attention', or 'weighted'
            alpha: Weight for attention in weighted method (default: 0.7)
            mil_model: Trained MIL model (required for 'attention' or 'weighted')
            device: Torch device (required for 'attention' or 'weighted')
            normalize_features: If True, L2-normalize features before ranking (default: True)
        """
        self.k = k
        self.ranking_method = ranking_method
        self.alpha = alpha
        self.mil_model = mil_model
        self.device = device
        self.normalize_features = normalize_features
        self.ranker = TileRanker()
        
        # Validate parameters for attention/weighted methods
        if ranking_method in ['attention', 'weighted']:
            if mil_model is None:
                raise ValueError(f"{ranking_method} method requires mil_model")
            if device is None:
                raise ValueError(f"{ranking_method} method requires device")
            if device.type != 'cuda':
                raise ValueError(f"{ranking_method} method requires CUDA device, got {device.type}")
        
        # Log normalization setting
        if normalize_features:
            logger.info("Feature normalization ENABLED (reproducible mode)")
        else:
            logger.warning("Feature normalization DISABLED (legacy mode)")
    
    def select_top_k_from_hdf5(
        self,
        input_path: str,
        output_path: str
    ) -> Dict:
        """
        Load HDF5, rank tiles, select top-K, save filtered HDF5.
        
        Args:
            input_path: Path to input HDF5 file
            output_path: Path to output filtered HDF5 file
        
        Returns:
            Dictionary with selection statistics
        """
        slide_name = Path(input_path).stem
        logger.info(f"Processing: {slide_name}")
        
        # Load features and coordinates
        features, coordinates, metadata = self._load_hdf5(input_path)
        original_num_tiles = len(features)
        
        logger.info(f"Loaded {original_num_tiles} tiles")
        
        # Rank tiles (returns dict with score arrays)
        score_dict = self._rank_tiles(features)
        
        # Select top-K based on final scores
        final_scores = score_dict['final_scores']
        top_k_indices = self.ranker.select_top_k(final_scores, self.k)
        selected_k = len(top_k_indices)
        
        # Filter features and coordinates
        filtered_features = features[top_k_indices]
        filtered_coords = coordinates[top_k_indices]
        
        # Filter all score types
        filtered_score_dict = {}
        for score_name, score_array in score_dict.items():
            filtered_score_dict[score_name] = score_array[top_k_indices]
        
        logger.info(f"Selected {selected_k} / {original_num_tiles} tiles ({selected_k/original_num_tiles:.1%})")
        
        # Compute score statistics (storage optimization)
        score_stats = self.ranker.get_score_stats(filtered_score_dict['final_scores'])
        
        # Update metadata
        output_metadata = metadata.copy()
        output_metadata['original_num_tiles'] = original_num_tiles
        output_metadata['selected_k'] = selected_k
        output_metadata['ranking_method'] = self.ranking_method
        output_metadata['reduction_ratio'] = 1 - (selected_k / original_num_tiles)
        output_metadata['score_stats'] = score_stats
        output_metadata['features_normalized'] = self.normalize_features  # Track normalization
        
        # Save filtered HDF5
        self._save_filtered_hdf5(
            features=filtered_features,
            coordinates=filtered_coords,
            score_dict=filtered_score_dict,
            metadata=output_metadata,
            output_path=output_path
        )
        
        logger.info(f"✓ Saved to {output_path}")
        
        # Return statistics
        stats = {
            'slide_name': slide_name,
            'original_num_tiles': original_num_tiles,
            'selected_k': selected_k,
            'reduction_ratio': output_metadata['reduction_ratio'],
            'score_stats': score_stats
        }
        
        return stats
    
    def _load_hdf5(self, path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load features, coordinates, and metadata from HDF5.
        
        Args:
            path: Path to HDF5 file
        
        Returns:
            Tuple of (features, coordinates, metadata)
        """
        with h5py.File(path, 'r') as f:
            features = f['features'][:]
            coordinates = f['coordinates'][:]
            
            # Load metadata
            metadata = {}
            for key in f['features'].attrs.keys():
                metadata[key] = f['features'].attrs[key]
            
            # Load coordinate metadata
            for key in f['coordinates'].attrs.keys():
                metadata[f'coord_{key}'] = f['coordinates'].attrs[key]
        
        return features, coordinates, metadata
    
    def _rank_tiles(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Rank tiles using specified method with optional normalization.
        
        Args:
            features: Feature array (N, feature_dim)
        
        Returns:
            Dictionary with score arrays:
            - 'final_scores': Scores used for ranking (N,)
            - 'attention_scores': Attention weights (N,) [if applicable]
            - 'l2_norm_scores': L2 norms (N,) [if applicable]
        """
        if self.ranking_method == 'feature_norm':
            final_scores = self.ranker.rank_by_feature_norm(features, normalize_first=self.normalize_features)
            return {'final_scores': final_scores}
        
        elif self.ranking_method == 'attention':
            attention_scores = self.ranker.rank_by_attention(
                features, self.mil_model, self.device
            )
            return {
                'final_scores': attention_scores,
                'attention_scores': attention_scores
            }
        
        elif self.ranking_method == 'weighted':
            final_scores, attention_scores, l2_norm_scores = \
                self.ranker.rank_by_weighted_combination(
                    features, self.mil_model, self.alpha, self.device
                )
            return {
                'final_scores': final_scores,
                'attention_scores': attention_scores,
                'l2_norm_scores': l2_norm_scores
            }
        
        else:
            raise ValueError(f"Unknown ranking method: {self.ranking_method}")
    
    def _save_filtered_hdf5(
        self,
        features: np.ndarray,
        coordinates: np.ndarray,
        score_dict: Dict[str, np.ndarray],
        metadata: Dict,
        output_path: str
    ):
        """
        Save filtered features to HDF5.
        
        Args:
            features: Filtered feature array (K, feature_dim)
            coordinates: Filtered coordinate array (K, 2)
            score_dict: Dictionary of score arrays
            metadata: Metadata dictionary
            output_path: Path to output HDF5 file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, 'w') as f:
            # Save features
            features_ds = f.create_dataset(
                'features',
                data=features,
                compression='gzip',
                compression_opts=4
            )
            
            # Save coordinates (preserving level-0 space)
            coords_ds = f.create_dataset(
                'coordinates',
                data=coordinates,
                compression='gzip',
                compression_opts=4
            )
            
            # Save score arrays as datasets
            if 'attention_scores' in score_dict:
                f.create_dataset(
                    'attention_scores',
                    data=score_dict['attention_scores'],
                    compression='gzip',
                    compression_opts=4
                )
            
            if 'l2_norm_scores' in score_dict:
                f.create_dataset(
                    'l2_norm_scores',
                    data=score_dict['l2_norm_scores'],
                    compression='gzip',
                    compression_opts=4
                )
            
            # Always save final scores
            f.create_dataset(
                'final_scores',
                data=score_dict['final_scores'],
                compression='gzip',
                compression_opts=4
            )
            
            # Add metadata to features dataset
            for key in ['slide_name', 'model_name', 'feature_dim', 'magnification']:
                if key in metadata:
                    features_ds.attrs[key] = metadata[key]
            
            # Add Top-K specific metadata
            features_ds.attrs['original_num_tiles'] = metadata['original_num_tiles']
            features_ds.attrs['selected_k'] = metadata['selected_k']
            features_ds.attrs['num_tiles'] = metadata['selected_k']  # Updated count
            features_ds.attrs['ranking_method'] = metadata['ranking_method']
            features_ds.attrs['sampling_method'] = self.ranking_method
            features_ds.attrs['reduction_ratio'] = metadata['reduction_ratio']
            features_ds.attrs['features_normalized'] = str(metadata.get('features_normalized', False))
            
            # Add alpha if using weighted method
            if self.ranking_method == 'weighted':
                features_ds.attrs['alpha'] = self.alpha
            
            # Add score statistics (storage optimized)
            for stat_key, stat_value in metadata['score_stats'].items():
                features_ds.attrs[f'score_{stat_key}'] = stat_value
            
            # Add coordinate metadata (preserve level-0 space)
            coords_ds.attrs['space'] = metadata.get('coord_space', 'level_0')
            coords_ds.attrs['description'] = 'Top-K selected tile coordinates in level-0 pixel space'
            
            if 'coord_extraction_level' in metadata:
                coords_ds.attrs['extraction_level'] = metadata['coord_extraction_level']
        
        logger.debug(f"Saved filtered HDF5: {features.shape} features, {coordinates.shape} coordinates")


if __name__ == '__main__':
    # Test TopKSelector
    print("Testing TopKSelector...")
    
    # Create dummy HDF5 file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_input:
        input_path = tmp_input.name
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_output:
        output_path = tmp_output.name
    
    try:
        # Create test HDF5
        features = np.random.randn(2000, 2048).astype(np.float32)
        coordinates = np.random.randint(0, 10000, (2000, 2)).astype(np.int32)
        
        with h5py.File(input_path, 'w') as f:
            features_ds = f.create_dataset('features', data=features)
            coords_ds = f.create_dataset('coordinates', data=coordinates)
            
            features_ds.attrs['slide_name'] = 'test_slide'
            features_ds.attrs['model_name'] = 'resnet50-imagenet'
            features_ds.attrs['feature_dim'] = 2048
            coords_ds.attrs['space'] = 'level_0'
        
        print(f"✓ Created test HDF5 with {len(features)} tiles")
        
        # Run Top-K selection
        selector = TopKSelector(k=1000, ranking_method='feature_norm')
        stats = selector.select_top_k_from_hdf5(input_path, output_path)
        
        print(f"✓ Top-K selection complete")
        print(f"  Original: {stats['original_num_tiles']}")
        print(f"  Selected: {stats['selected_k']}")
        print(f"  Reduction: {stats['reduction_ratio']:.1%}")
        
        # Verify output
        with h5py.File(output_path, 'r') as f:
            assert f['features'].shape == (1000, 2048)
            assert f['coordinates'].shape == (1000, 2)
            assert f['coordinates'].attrs['space'] == 'level_0'
            assert 'ranking_method' in f['features'].attrs
            assert 'score_min' in f['features'].attrs
            print(f"✓ Output HDF5 verified")
        
        print("\n✅ TopKSelector tests passed!")
    
    finally:
        # Cleanup
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
