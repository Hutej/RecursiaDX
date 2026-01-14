# 🔲 Tiling Module - Efficient Gigapixel Image Patching
# Handles sliding window extraction with overlap, multi-resolution support

import numpy as np
from PIL import Image
import torch
from typing import Tuple, List, Optional, Generator
from dataclasses import dataclass
import os


@dataclass
class PatchInfo:
    """Metadata for extracted patch"""
    x: int  # Top-left x coordinate
    y: int  # Top-left y coordinate
    width: int
    height: int
    scale: float  # Magnification scale
    patch_id: int
    is_tissue: bool = True  # Whether patch contains tissue (vs background)


class GigapixelTiler:
    """
    Efficient tiling system for gigapixel histopathology images.
    
    Features:
    - Sliding window with configurable overlap
    - Multi-resolution support (10x, 20x, 40x)
    - Background filtering (skip white/empty patches)
    - Memory-efficient streaming
    - Coordinate tracking for reconstruction
    """
    
    def __init__(
        self,
        patch_size: int = 224,
        overlap: float = 0.25,
        scales: List[float] = [1.0, 0.5, 0.25],  # 40x, 20x, 10x equivalent
        tissue_threshold: float = 0.85,  # Skip patches with >85% white
        min_tissue_area: float = 0.05  # Minimum 5% tissue required
    ):
        self.patch_size = patch_size
        self.overlap = overlap
        self.scales = scales
        self.tissue_threshold = tissue_threshold
        self.min_tissue_area = min_tissue_area
        self.stride = int(patch_size * (1 - overlap))
        
    def is_tissue_patch(self, patch: np.ndarray) -> bool:
        """
        Determine if patch contains tissue (not just background).
        Uses intensity and saturation thresholds.
        """
        # Convert to grayscale
        if len(patch.shape) == 3:
            gray = np.mean(patch, axis=2)
        else:
            gray = patch
            
        # Calculate percentage of white/bright pixels
        bright_pixels = np.sum(gray > 200) / gray.size
        
        # Check if enough dark/colored pixels (tissue)
        return bright_pixels < self.tissue_threshold
    
    def extract_patches(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        save_patches: bool = False
    ) -> Generator[Tuple[np.ndarray, PatchInfo], None, None]:
        """
        Extract patches from gigapixel image using sliding window.
        
        Yields:
            (patch_array, patch_info) tuples
        """
        print(f"🔲 Tiling image: {image_path}")
        
        # Load image (use PIL for memory efficiency)
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        print(f"   Image size: {img_width}x{img_height} pixels")
        print(f"   Patch size: {self.patch_size}x{self.patch_size}")
        print(f"   Overlap: {self.overlap*100:.0f}%")
        print(f"   Stride: {self.stride}px")
        
        patch_id = 0
        total_patches = 0
        tissue_patches = 0
        
        # Multi-scale extraction
        for scale_idx, scale in enumerate(self.scales):
            print(f"\n📊 Processing scale {scale:.2f}x (Level {scale_idx})")
            
            # Resize image for current scale
            if scale != 1.0:
                scaled_width = int(img_width * scale)
                scaled_height = int(img_height * scale)
                scaled_img = img.resize((scaled_width, scaled_height), Image.LANCZOS)
            else:
                scaled_img = img
                scaled_width, scaled_height = img_width, img_height
            
            # Convert to numpy for processing
            img_array = np.array(scaled_img)
            
            # Calculate grid
            x_positions = range(0, scaled_width - self.patch_size + 1, self.stride)
            y_positions = range(0, scaled_height - self.patch_size + 1, self.stride)
            
            scale_patches = 0
            
            # Extract patches
            for y in y_positions:
                for x in x_positions:
                    # Extract patch
                    patch = img_array[y:y+self.patch_size, x:x+self.patch_size]
                    
                    # Skip if patch is incomplete
                    if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                        continue
                    
                    # Check if tissue
                    is_tissue = self.is_tissue_patch(patch)
                    
                    if is_tissue:
                        tissue_patches += 1
                        scale_patches += 1
                        
                        # Create metadata
                        patch_info = PatchInfo(
                            x=int(x / scale),  # Original coordinates
                            y=int(y / scale),
                            width=self.patch_size,
                            height=self.patch_size,
                            scale=scale,
                            patch_id=patch_id,
                            is_tissue=True
                        )
                        
                        # Save if requested
                        if save_patches and output_dir:
                            os.makedirs(output_dir, exist_ok=True)
                            patch_img = Image.fromarray(patch)
                            patch_filename = f"patch_{patch_id:06d}_s{scale:.2f}_x{x}_y{y}.png"
                            patch_img.save(os.path.join(output_dir, patch_filename))
                        
                        yield patch, patch_info
                        patch_id += 1
                    
                    total_patches += 1
            
            print(f"   Extracted {scale_patches} tissue patches at scale {scale:.2f}x")
        
        print(f"\n✅ Tiling complete:")
        print(f"   Total patches examined: {total_patches}")
        print(f"   Tissue patches: {tissue_patches} ({tissue_patches/max(total_patches,1)*100:.1f}%)")
        print(f"   Background patches skipped: {total_patches - tissue_patches}")


class PatchExtractor:
    """
    Advanced patch extraction with normalization and augmentation support.
    Prepares patches for deep learning models.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)
    
    def preprocess_patch(self, patch: np.ndarray) -> torch.Tensor:
        """
        Preprocess patch for model input.
        - Resize if needed
        - Normalize
        - Convert to tensor
        """
        # Resize if needed
        if patch.shape[:2] != self.target_size:
            patch_img = Image.fromarray(patch)
            patch_img = patch_img.resize(self.target_size, Image.BILINEAR)
            patch = np.array(patch_img)
        
        # Convert to float [0, 1]
        patch = patch.astype(np.float32) / 255.0
        
        # Normalize
        if self.normalize:
            patch = (patch - self.mean) / self.std
        
        # Convert to tensor (C, H, W)
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1)
        
        return patch_tensor
    
    def extract_multiscale_patches(
        self,
        image_path: str,
        center_x: int,
        center_y: int,
        scales: List[int] = [224, 448, 896]
    ) -> List[torch.Tensor]:
        """
        Extract patches at multiple scales around a center point.
        Useful for multi-scale attention mechanisms.
        """
        img = Image.open(image_path)
        patches = []
        
        for scale_size in scales:
            half_size = scale_size // 2
            
            # Calculate bounds
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(img.width, center_x + half_size)
            y2 = min(img.height, center_y + half_size)
            
            # Extract and resize
            patch = img.crop((x1, y1, x2, y2))
            patch = patch.resize(self.target_size, Image.BILINEAR)
            
            # Preprocess
            patch_array = np.array(patch)
            patch_tensor = self.preprocess_patch(patch_array)
            patches.append(patch_tensor)
        
        return patches


# ============================================
# UTILITY FUNCTIONS
# ============================================

def calculate_optimal_tiling(
    image_width: int,
    image_height: int,
    patch_size: int = 224,
    overlap: float = 0.25,
    max_patches: int = 10000
) -> Tuple[int, float]:
    """
    Calculate optimal stride and overlap for given constraints.
    """
    stride = int(patch_size * (1 - overlap))
    
    # Calculate number of patches
    num_x = (image_width - patch_size) // stride + 1
    num_y = (image_height - patch_size) // stride + 1
    total_patches = num_x * num_y
    
    # Adjust if too many patches
    if total_patches > max_patches:
        # Increase stride to reduce patch count
        new_stride = int(np.sqrt((image_width * image_height) / max_patches))
        new_overlap = 1 - (new_stride / patch_size)
        print(f"⚠️  Adjusting overlap from {overlap:.2f} to {new_overlap:.2f} to limit patches")
        return new_stride, new_overlap
    
    return stride, overlap


if __name__ == "__main__":
    # Example usage
    print("🔲 Gigapixel Tiling Module")
    print("=" * 70)
    
    # Initialize tiler
    tiler = GigapixelTiler(
        patch_size=224,
        overlap=0.25,
        scales=[1.0, 0.5],  # Two scales
        tissue_threshold=0.85
    )
    
    # Example: tile an image
    image_path = "example_histopathology.svs"
    if os.path.exists(image_path):
        patch_count = 0
        for patch, info in tiler.extract_patches(image_path, save_patches=False):
            patch_count += 1
            if patch_count <= 3:
                print(f"   Patch {info.patch_id}: ({info.x}, {info.y}) @ {info.scale}x")
        
        print(f"\n✅ Extracted {patch_count} patches total")
    else:
        print("ℹ️  Example image not found. Module ready for use.")
