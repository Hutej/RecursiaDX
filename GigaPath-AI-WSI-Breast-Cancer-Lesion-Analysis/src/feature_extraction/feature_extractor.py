"""
Feature extraction pipeline with HDF5 caching.
Extracts embeddings from WSI tiles using frozen backbone.
"""
import torch
import torch.nn as nn
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator
from tqdm import tqdm

from .backbones import BackboneLoader
from ..preprocessing import TileExtractor
from ..utils.logger import get_logger
from ..utils.gpu_monitor import GPUMonitor, get_device

logger = get_logger(__name__)


class FeatureExtractor:
    """Extract and cache features from WSI tiles."""
    
    def __init__(
        self,
        model_name: str = 'resnet50-imagenet',
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = True
    ):
        """
        Initialize feature extractor.
        
        Args:
            model_name: Backbone model name
            batch_size: Batch size (if None, auto-calculate)
            device: Torch device (if None, auto-detect)
            use_amp: Use automatic mixed precision
        """
        self.model_name = model_name
        self.use_amp = use_amp
        
        # Setup device (Force GPU)
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA NOT AVAILABLE — GPU REQUIRED")
            self.device = torch.device("cuda")
        else:
            if device.type != 'cuda':
                 raise RuntimeError("CPU DEVICE DISALLOWED — GPU REQUIRED")
            self.device = device
        
        # Load backbone model
        self.model, self.feature_dim = BackboneLoader.load_backbone(
            model_name=model_name,
            freeze=True,
            device=self.device
        )
        
        # Get transforms
        self.transforms = BackboneLoader.get_input_transforms(model_name)
        
        # Setup GPU monitor
        self.gpu_monitor = GPUMonitor() if torch.cuda.is_available() else None
        
        # Auto-calculate batch size if None
        if batch_size is None:
            self.batch_size = self._calculate_safe_batch_size()
        else:
            self.batch_size = batch_size
        
        logger.info(f"FeatureExtractor initialized")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Feature dim: {self.feature_dim}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  AMP: {self.use_amp}")
    
    def _calculate_safe_batch_size(self) -> int:
        """Calculate safe batch size for GPU."""
        if not torch.cuda.is_available():
             raise RuntimeError("CUDA NOT AVAILABLE — GPU REQUIRED")
        
        # Estimate memory per tile
        # 256x256x3 image → ~0.75 MB as float32
        # ResNet50 forward pass → ~2-3x overhead
        
        max_vram_gb = 7.5  # RTX 4070 with margin
        model_overhead_gb = 1.5  # Model weights
        tile_overhead_gb = 0.003  # ~3 MB per tile (input + activations)
        
        available_gb = max_vram_gb - model_overhead_gb
        batch_size = int(available_gb / tile_overhead_gb)
        
        # Clamp to reasonable range
        batch_size = max(16, min(batch_size, 128))
        
        logger.info(f"Auto-calculated batch size: {batch_size}")
        
        return batch_size
    
    def extract_features_from_slide(
        self,
        slide_path: str,
        output_path: str,
        tile_extractor: Optional[TileExtractor] = None,
        **tile_extractor_kwargs
    ) -> Dict:
        """
        Extract features from a single WSI and save to HDF5.
        
        Args:
            slide_path: Path to WSI file
            output_path: Path to save HDF5 file
            tile_extractor: TileExtractor instance (if None, create new)
            **tile_extractor_kwargs: Arguments for TileExtractor
        
        Returns:
            Dictionary with extraction statistics
        """
        slide_name = Path(slide_path).stem
        logger.info(f"Extracting features from: {slide_name}")
        
        # Create tile extractor if not provided
        if tile_extractor is None:
            tile_extractor = TileExtractor(
                save_tiles=False,  # Production mode - transient extraction
                **tile_extractor_kwargs
            )
        
        # Extract features
        features_list = []
        coords_list = []
        metadata_list = []
        
        # Get tile generator
        tile_generator = tile_extractor.extract_tiles_generator(slide_path)
        
        # Process in batches
        batch_tiles = []
        batch_metadata = []
        
        for tile_array, tile_metadata in tqdm(tile_generator, desc=f"Extracting {slide_name}"):
            # Apply transforms
            tile_tensor = self.transforms(tile_array)
            batch_tiles.append(tile_tensor)
            batch_metadata.append(tile_metadata)
            
            # Process batch when full
            if len(batch_tiles) >= self.batch_size:
                batch_features = self._extract_batch(batch_tiles)
                features_list.extend(batch_features)
                coords_list.extend([(m['x'], m['y']) for m in batch_metadata])
                metadata_list.extend(batch_metadata)
                
                # Clear batch
                batch_tiles = []
                batch_metadata = []
        
        # Process remaining tiles
        if len(batch_tiles) > 0:
            batch_features = self._extract_batch(batch_tiles)
            features_list.extend(batch_features)
            coords_list.extend([(m['x'], m['y']) for m in batch_metadata])
            metadata_list.extend(batch_metadata)
        
        # Convert to arrays
        features_array = np.array(features_list, dtype=np.float32)
        coords_array = np.array(coords_list, dtype=np.int32)
        
        logger.info(f"Extracted {len(features_list)} tile features")
        logger.info(f"Features shape: {features_array.shape}")
        logger.info(f"Coordinates shape: {coords_array.shape}")
        
        # Save to HDF5
        self._save_features_hdf5(
            features=features_array,
            coordinates=coords_array,
            metadata=metadata_list[0] if metadata_list else {},  # Use first tile metadata
            output_path=output_path,
            slide_name=slide_name
        )
        
        # Return statistics
        stats = {
            'slide_name': slide_name,
            'num_tiles': len(features_list),
            'feature_dim': self.feature_dim,
            'features_shape': features_array.shape,
            'coordinates_shape': coords_array.shape
        }
        
        logger.info(f"[OK] Features saved to {output_path}")
        
        return stats
    
    def extract_features_from_tiles(
        self,
        tiles_dir: Path,
        coordinates_csv: Path,
        output_path: Path,
        slide_name: str
    ) -> Dict:
        """
        Extract features from preprocessed tiles saved to disk.
        
        Args:
            tiles_dir: Directory containing PNG tiles
            coordinates_csv: Path to coordinates.csv file
            output_path: Path to save HDF5 file
            slide_name: Name of the slide
        
        Returns:
            Dictionary with extraction statistics
        """
        import pandas as pd
        from PIL import Image
        
        logger.info(f"Extracting features from preprocessed tiles: {slide_name}")
        
        # Load coordinates
        coords_df = pd.read_csv(coordinates_csv)
        logger.info(f"Loaded {len(coords_df)} tile coordinates")
        
        # Get list of tile files
        tile_files = sorted(list(Path(tiles_dir).glob("*.png")))
        logger.info(f"Found {len(tile_files)} PNG tiles")
        
        if len(tile_files) == 0:
            raise ValueError(f"No PNG tiles found in {tiles_dir}")
        
        # Extract features
        features_list = []
        coords_list = []
        
        # Process in batches
        batch_tiles = []
        batch_coords = []
        
        for tile_file in tqdm(tile_files, desc=f"Processing {slide_name}"):
            # Parse coordinates from filename (e.g., "tile_1024_2048.png")
            filename = tile_file.stem
            parts = filename.split('_')
            if len(parts) >= 3:
                x_coord = int(parts[1])
                y_coord = int(parts[2])
            else:
                logger.warning(f"Skipping tile with unexpected filename format: {filename}")
                continue
            
            try:
                # Load tile image
                tile_img = Image.open(tile_file).convert('RGB')
                tile_array = np.array(tile_img)
                
                # Apply transforms
                tile_tensor = self.transforms(tile_array)
                batch_tiles.append(tile_tensor)
                batch_coords.append((x_coord, y_coord))
                
                # Process batch when full
                if len(batch_tiles) >= self.batch_size:
                    batch_features = self._extract_batch(batch_tiles)
                    features_list.extend(batch_features)
                    coords_list.extend(batch_coords)
                    
                    # Clear batch
                    batch_tiles = []
                    batch_coords = []
            
            except Exception as e:
                logger.warning(f"Skipping corrupted tile {filename}: {e}")
                continue
        
        # Process remaining tiles
        if len(batch_tiles) > 0:
            batch_features = self._extract_batch(batch_tiles)
            features_list.extend(batch_features)
            coords_list.extend(batch_coords)
        
        # Convert to arrays
        features_array = np.array(features_list, dtype=np.float32)
        coords_array = np.array(coords_list, dtype=np.int32)
        
        logger.info(f"Extracted {len(features_list)} tile features")
        logger.info(f"Features shape: {features_array.shape}")
        logger.info(f"Coordinates shape: {coords_array.shape}")
        
        # Save to HDF5
        self._save_features_hdf5(
            features=features_array,
            coordinates=coords_array,
            metadata={},  # Metadata could be loaded from metadata.json if needed
            output_path=str(output_path),
            slide_name=slide_name
        )
        
        # Return statistics
        stats = {
            'slide_name': slide_name,
            'num_tiles': len(features_list),
            'feature_dim': self.feature_dim,
            'features_shape': features_array.shape,
            'coordinates_shape': coords_array.shape
        }
        
        logger.info(f"[OK] Features saved to {output_path}")
        
        return stats
    
    def _extract_batch(self, batch_tiles: List[torch.Tensor]) -> List[np.ndarray]:
        """
        Extract features from a batch of tiles with enforced AMP and inference mode.
        
        Args:
            batch_tiles: List of tile tensors
        
        Returns:
            List of feature arrays
        """
        # Stack into batch
        batch = torch.stack(batch_tiles).to(self.device)
        
        # ENFORCED: AMP + Inference Mode (no gradients, memory efficient)
        with torch.inference_mode():  # Guaranteed no grad tracking
            if self.use_amp and torch.cuda.is_available():
                # Enforced automatic mixed precision
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    features = self.model(batch)
            else:
                # Fallback to float32 (should not happen with GPU enforcement)
                features = self.model(batch)
        
        # Convert to numpy float32 for consistency
        features_np = features.cpu().float().numpy()
        
        return list(features_np)
    
    def _save_features_hdf5(
        self,
        features: np.ndarray,
        coordinates: np.ndarray,
        metadata: Dict,
        output_path: str,
        slide_name: str
    ):
        """
        Save features to HDF5 file with proper structure and metadata.
        
        Args:
            features: Feature array (N, feature_dim)
            coordinates: Coordinate array (N, 2) - (x, y) in level-0 space
            metadata: Tile metadata dictionary
            output_path: Path to HDF5 file
            slide_name: Name of slide
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
            
            # Save coordinates (ALWAYS in level-0 space)
            coords_ds = f.create_dataset(
                'coordinates',
                data=coordinates,
                compression='gzip',
                compression_opts=4
            )
            
            # Add metadata attributes to features dataset
            features_ds.attrs['slide_name'] = slide_name
            features_ds.attrs['model_name'] = self.model_name
            features_ds.attrs['feature_dim'] = self.feature_dim
            features_ds.attrs['num_tiles'] = len(features)
            
            # Add coordinate space metadata (CRITICAL for explainability)
            coords_ds.attrs['space'] = 'level_0'  # ALWAYS level-0
            coords_ds.attrs['description'] = 'Tile coordinates in level-0 (highest resolution) pixel space'
            
            # Add extraction metadata if available
            if metadata:
                if 'magnification' in metadata:
                    features_ds.attrs['magnification'] = metadata['magnification']
                if 'level' in metadata:
                    coords_ds.attrs['extraction_level'] = metadata['level']
        
        logger.debug(f"HDF5 file created: {output_file}")
        logger.debug(f"  Features: {features.shape} (compressed)")
        logger.debug(f"  Coordinates: {coordinates.shape} (level-0 space)")
    
    @staticmethod
    def load_features_hdf5(hdf5_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load features from HDF5 file.
        
        Args:
            hdf5_path: Path to HDF5 file
        
        Returns:
            Tuple of (features, coordinates, metadata)
        """
        with h5py.File(hdf5_path, 'r') as f:
            features = f['features'][:]
            coordinates = f['coordinates'][:]
            
            # Load metadata
            metadata = {
                'slide_name': f['features'].attrs.get('slide_name', ''),
                'model_name': f['features'].attrs.get('model_name', ''),
                'feature_dim': f['features'].attrs.get('feature_dim', 0),
                'num_tiles': f['features'].attrs.get('num_tiles', 0),
                'coordinate_space': f['coordinates'].attrs.get('space', 'unknown'),
                'magnification': f['features'].attrs.get('magnification', None)
            }
        
        return features, coordinates, metadata


if __name__ == '__main__':
    # Test feature extractor
    print("Testing FeatureExtractor...")
    
    extractor = FeatureExtractor(model_name='resnet50-imagenet')
    
    # Test single tile
    dummy_tile = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    tile_tensor = extractor.transforms(dummy_tile).unsqueeze(0).to(extractor.device)
    
    with torch.no_grad():
        features = extractor.model(tile_tensor)
    
    print(f"✓ Input: {dummy_tile.shape}")
    print(f"✓ Output: {features.shape}")
    print(f"✓ Expected: (1, {extractor.feature_dim})")
    
    assert features.shape == (1, extractor.feature_dim)
    print("\n✅ FeatureExtractor test passed!")
