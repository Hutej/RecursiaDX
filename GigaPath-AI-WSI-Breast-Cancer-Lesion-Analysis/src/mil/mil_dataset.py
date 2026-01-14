"""
PyTorch Dataset for MIL training.
Loads HDF5 feature files with slide-level labels.
"""
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MILDataset(Dataset):
    """
    MIL Dataset for loading HDF5 features with slide-level labels.
    
    Each item is a "bag" (WSI slide) containing K instances (tile features).
    """
    
    def __init__(
        self,
        hdf5_dir: str,
        labels_csv: str,
        feature_key: str = 'features',
        coord_key: str = 'coordinates'
    ):
        """
        Initialize MIL dataset.
        
        Args:
            hdf5_dir: Directory containing HDF5 feature files
            labels_csv: Path to labels CSV (columns: slide_name, label)
            feature_key: Key for features in HDF5
            coord_key: Key for coordinates in HDF5
        """
        self.hdf5_dir = Path(hdf5_dir)
        self.feature_key = feature_key
        self.coord_key = coord_key
        
        # Load labels
        labels_df = pd.read_csv(labels_csv)
        
        # Filter to slides that exist in hdf5_dir
        self.slide_data = []
        for _, row in labels_df.iterrows():
            slide_name = row['slide_name']
            label = int(row['label'])
            
            # Look for HDF5 file
            h5_path = self.hdf5_dir / f"{slide_name}.h5"
            if not h5_path.exists():
                # Try with _topk suffix
                h5_path = self.hdf5_dir / f"{slide_name}_topk.h5"
            
            if h5_path.exists():
                self.slide_data.append({
                    'slide_name': slide_name,
                    'h5_path': h5_path,
                    'label': label
                })
            else:
                logger.warning(f"HDF5 file not found for slide: {slide_name}")
        
        logger.info(f"MILDataset initialized with {len(self.slide_data)} slides")
    
    def __len__(self) -> int:
        return len(self.slide_data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single MIL bag (slide).
        
        Args:
            idx: Index of slide
        
        Returns:
            Dictionary with features, label, coords, slide_name
        """
        slide_info = self.slide_data[idx]
        
        # Load HDF5
        with h5py.File(slide_info['h5_path'], 'r') as f:
            features = f[self.feature_key][:]  # (K, feature_dim)
            coords = f[self.coord_key][:]      # (K, 2)
        
        # Convert to tensors
        features_tensor = torch.from_numpy(features).float()
        coords_tensor = torch.from_numpy(coords).long()
        label_tensor = torch.tensor(slide_info['label']).long()
        
        return {
            'features': features_tensor,     # (K, feature_dim)
            'coords': coords_tensor,         # (K, 2)
            'label': label_tensor,           # ()
            'slide_name': slide_info['slide_name']
        }
    
    def get_labels(self) -> List[int]:
        """Get all slide labels for stratification."""
        return [slide['label'] for slide in self.slide_data]
    
    @staticmethod
    def collate_mil(batch: List[Dict]) -> Dict:
        """
        Custom collate function for MIL.
        
        Note: Batch size should be 1 for standard MIL.
        Handles variable-length bags.
        
        Args:
            batch: List of dictionaries from __getitem__
        
        Returns:
            Batched dictionary
        """
        if len(batch) != 1:
            logger.warning(
                f"MIL typically uses batch_size=1. Got batch_size={len(batch)}. "
                f"Each slide has different K, cannot stack."
            )
        
        # For batch_size=1, just return the single item
        if len(batch) == 1:
            return batch[0]
        
        # For batch_size>1, return list (not ideal for MIL)
        return {
            'features': [item['features'] for item in batch],
            'coords': [item['coords'] for item in batch],
            'label': torch.stack([item['label'] for item in batch]),
            'slide_name': [item['slide_name'] for item in batch]
        }


if __name__ == '__main__':
    # Test MILDataset
    print("Testing MILDataset...")
    
    import tempfile
    import os
    
    # Create temp directory and files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy HDF5 files
        for i in range(5):
            h5_path = Path(tmpdir) / f"slide_{i}.h5"
            features = np.random.randn(1000, 2048).astype(np.float32)
            coords = np.random.randint(0, 10000, (1000, 2)).astype(np.int32)
            
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('features', data=features)
                f.create_dataset('coordinates', data=coords)
        
        # Create labels CSV
        labels_csv = Path(tmpdir) / 'labels.csv'
        labels_df = pd.DataFrame({
            'slide_name': [f'slide_{i}' for i in range(5)],
            'label': [0, 1, 0, 1, 0]
        })
        labels_df.to_csv(labels_csv, index=False)
        
        print(f"✓ Created test data in {tmpdir}")
        
        # Create dataset
        dataset = MILDataset(hdf5_dir=tmpdir, labels_csv=str(labels_csv))
        print(f"✓ Dataset created with {len(dataset)} slides")
        
        # Test __getitem__
        sample = dataset[0]
        print(f"✓ Sample loaded:")
        print(f"  Features: {sample['features'].shape}")
        print(f"  Coords: {sample['coords'].shape}")
        print(f"  Label: {sample['label'].item()}")
        print(f"  Slide: {sample['slide_name']}")
        
        # Test collate function
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=1, collate_fn=MILDataset.collate_mil)
        batch = next(iter(loader))
        print(f"✓ DataLoader test:")
        print(f"  Features: {batch['features'].shape}")
        print(f"  Label: {batch['label'].item()}")
        
        print("\n✅ MILDataset tests passed!")
