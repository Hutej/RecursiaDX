"""
Tissue detection module for WSI preprocessing.
Separates tissue regions from background using Otsu thresholding.
"""
import numpy as np
import cv2
from typing import Tuple, Optional
from PIL import Image

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TissueDetector:
    """Detect tissue regions in whole slide images."""
    
    def __init__(
        self,
        thumbnail_size: int = 2000,
        gray_threshold: int = 200,
        morph_kernel_close: int = 10,
        morph_kernel_open: int = 5
    ):
        """
        Initialize tissue detector.
        
        Args:
            thumbnail_size: Size of thumbnail for tissue detection
            gray_threshold: Grayscale threshold for background (higher = lighter background)
            morph_kernel_close: Kernel size for morphological closing
            morph_kernel_open: Kernel size for morphological opening
        """
        self.thumbnail_size = thumbnail_size
        self.gray_threshold = gray_threshold
        self.morph_kernel_close = morph_kernel_close
        self.morph_kernel_open = morph_kernel_open
    
    def detect_tissue(
        self,
        slide,
        thumbnail_size: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Detect tissue regions in WSI using Otsu thresholding.
        
        Args:
            slide: OpenSlide object
            thumbnail_size: Override default thumbnail size
        
        Returns:
            Tuple of (tissue_mask, tissue_coverage_ratio)
        """
        thumb_size = thumbnail_size or self.thumbnail_size
        
        # Get thumbnail
        thumbnail = self._get_thumbnail(slide, thumb_size)
        
        # Generate tissue mask
        tissue_mask = self._create_tissue_mask(thumbnail)
        
        # Calculate tissue coverage
        tissue_ratio = np.sum(tissue_mask) / tissue_mask.size
        
        logger.info(f"Tissue detection complete. Coverage: {tissue_ratio:.2%}")
        
        return tissue_mask, tissue_ratio
    
    def _get_thumbnail(self, slide, size: int) -> np.ndarray:
        """
        Get thumbnail from slide.
        
        Args:
            slide: OpenSlide object
            size: Maximum dimension size
        
        Returns:
            Thumbnail as numpy array (RGB)
        """
        # Get slide dimensions at level 0
        width, height = slide.dimensions
        
        # Calculate thumbnail size maintaining aspect ratio
        if width > height:
            thumb_width = size
            thumb_height = int(size * height / width)
        else:
            thumb_width = int(size * width / height)
            thumb_height = size
        
        # Get thumbnail from OpenSlide
        thumbnail = slide.get_thumbnail((thumb_width, thumb_height))
        
        # Convert PIL to numpy
        thumbnail_np = np.array(thumbnail)
        
        logger.debug(f"Thumbnail created: {thumbnail_np.shape}")
        
        return thumbnail_np
    
    def _create_tissue_mask(self, thumbnail: np.ndarray) -> np.ndarray:
        """
        Create binary tissue mask using Otsu thresholding.
        
        Args:
            thumbnail: RGB thumbnail image
        
        Returns:
            Binary tissue mask (True = tissue, False = background)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
        
        # Invert if needed (tissue is usually darker than background)
        # For H&E slides, background is typically white/light
        inverted = 255 - gray
        
        # Apply Otsu thresholding
        _, binary = cv2.threshold(
            inverted,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Additional filter: remove very light regions (likely background)
        binary[gray > self.gray_threshold] = 0
        
        # Morphological operations to clean mask
        # Closing: fill small holes in tissue
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_close, self.morph_kernel_close)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        
        # Opening: remove small noise
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_open, self.morph_kernel_open)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        
        # Convert to boolean mask
        tissue_mask = binary > 0
        
        return tissue_mask
    
    def calculate_tissue_ratio(
        self,
        tile_coords: Tuple[int, int],
        tile_size: int,
        tissue_mask: np.ndarray,
        slide_dimensions: Tuple[int, int]
    ) -> float:
        """
        Calculate tissue ratio for a specific tile.
        
        Args:
            tile_coords: (x, y) coordinates of tile in level 0
            tile_size: Size of tile in pixels
            tissue_mask: Binary tissue mask (from thumbnail)
            slide_dimensions: (width, height) of slide at level 0
        
        Returns:
            Tissue ratio (0.0 to 1.0)
        """
        x, y = tile_coords
        slide_w, slide_h = slide_dimensions
        
        # Map tile coordinates to thumbnail coordinates
        thumb_h, thumb_w = tissue_mask.shape
        
        # Calculate scale factors
        scale_x = thumb_w / slide_w
        scale_y = thumb_h / slide_h
        
        # Map tile to thumbnail space
        thumb_x = int(x * scale_x)
        thumb_y = int(y * scale_y)
        thumb_w_tile = int(tile_size * scale_x)
        thumb_h_tile = int(tile_size * scale_y)
        
        # Ensure within bounds
        thumb_x = max(0, min(thumb_x, thumb_w - 1))
        thumb_y = max(0, min(thumb_y, thumb_h - 1))
        thumb_x_end = min(thumb_x + thumb_w_tile, thumb_w)
        thumb_y_end = min(thumb_y + thumb_h_tile, thumb_h)
        
        # Extract tile region from mask
        tile_mask = tissue_mask[thumb_y:thumb_y_end, thumb_x:thumb_x_end]
        
        # Calculate tissue ratio
        if tile_mask.size == 0:
            return 0.0
        
        tissue_ratio = np.sum(tile_mask) / tile_mask.size
        
        return tissue_ratio
    
    def save_tissue_mask(
        self,
        tissue_mask: np.ndarray,
        output_path: str
    ):
        """
        Save tissue mask as image for visualization.
        
        Args:
            tissue_mask: Binary tissue mask
            output_path: Path to save mask image
        """
        # Convert boolean to uint8
        mask_img = (tissue_mask * 255).astype(np.uint8)
        
        # Save as PNG
        Image.fromarray(mask_img).save(output_path)
        
        logger.debug(f"Tissue mask saved to {output_path}")
