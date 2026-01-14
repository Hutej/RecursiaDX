"""
WSI utility functions for slide handling and coordinate mapping.
"""
import openslide
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


def open_slide(slide_path: str) -> openslide.OpenSlide:
    """
    Safely open a whole slide image.
    
    Args:
        slide_path: Path to WSI file
    
    Returns:
        OpenSlide object
    
    Raises:
        FileNotFoundError: If slide file doesn't exist
        openslide.OpenSlideError: If file is not a valid WSI
    """
    path = Path(slide_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Slide not found: {slide_path}")
    
    try:
        slide = openslide.OpenSlide(str(path))
        logger.info(f"Opened slide: {path.name}")
        return slide
    except openslide.OpenSlideError as e:
        logger.error(f"Failed to open slide {path.name}: {e}")
        raise


def get_slide_properties(slide: openslide.OpenSlide) -> Dict[str, Any]:
    """
    Extract slide properties and metadata.
    
    Args:
        slide: OpenSlide object
    
    Returns:
        Dictionary of slide properties
    """
    props = {
        'dimensions': slide.dimensions,  # (width, height) at level 0
        'level_count': slide.level_count,
        'level_dimensions': slide.level_dimensions,
        'level_downsamples': slide.level_downsamples,
        'vendor': slide.properties.get(openslide.PROPERTY_NAME_VENDOR, 'Unknown'),
    }
    
    # Try to get magnification
    mag = get_magnification(slide)
    if mag is not None:
        props['magnification'] = mag
    
    # Try to get microns per pixel (MPP)
    mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
    if mpp_x and mpp_y:
        props['mpp_x'] = float(mpp_x)
        props['mpp_y'] = float(mpp_y)
    
    return props


def get_magnification(slide: openslide.OpenSlide) -> Optional[float]:
    """
    Get slide magnification from properties.
    
    Args:
        slide: OpenSlide object
    
    Returns:
        Magnification value or None if not available
    """
    # Try objective power property
    mag_str = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    
    if mag_str:
        try:
            return float(mag_str)
        except ValueError:
            logger.warning(f"Invalid magnification value: {mag_str}")
    
    # Try vendor-specific properties
    vendor = slide.properties.get(openslide.PROPERTY_NAME_VENDOR, '').lower()
    
    if 'aperio' in vendor:
        mag_str = slide.properties.get('aperio.AppMag')
    elif 'hamamatsu' in vendor:
        mag_str = slide.properties.get('hamamatsu.SourceLens')
    else:
        mag_str = None
    
    if mag_str:
        try:
            return float(mag_str)
        except ValueError:
            pass
    
    logger.warning("Magnification not found in slide properties")
    return None


def get_best_level_for_magnification(
    slide: openslide.OpenSlide,
    target_magnification: float,
    tolerance: float = 0.1
) -> Tuple[int, float]:
    """
    Find the best pyramid level for target magnification.
    
    Args:
        slide: OpenSlide object
        target_magnification: Desired magnification (e.g., 20 for 20×)
        tolerance: Acceptable deviation (0.1 = ±10%)
    
    Returns:
        Tuple of (best_level, actual_magnification)
    """
    base_mag = get_magnification(slide)
    
    if base_mag is None:
        logger.warning("Base magnification unknown, using level 0")
        return 0, target_magnification
    
    # Calculate magnification at each level
    best_level = 0
    best_mag = base_mag
    min_diff = abs(base_mag - target_magnification)
    
    for level, downsample in enumerate(slide.level_downsamples):
        level_mag = base_mag / downsample
        diff = abs(level_mag - target_magnification)
        
        if diff < min_diff:
            min_diff = diff
            best_level = level
            best_mag = level_mag
    
    # Check if within tolerance
    mag_error = abs(best_mag - target_magnification) / target_magnification
    if mag_error > tolerance:
        logger.warning(
            f"Closest magnification {best_mag:.1f}× exceeds tolerance "
            f"(target: {target_magnification}×, error: {mag_error:.1%})"
        )
    
    logger.info(f"Using level {best_level} (magnification: {best_mag:.1f}×)")
    
    return best_level, best_mag


def get_tile_coordinates(
    slide_dimensions: Tuple[int, int],
    tile_size: int,
    overlap: int = 0
) -> list:
    """
    Generate grid of tile coordinates for a slide.
    
    Args:
        slide_dimensions: (width, height) of slide at level 0
        tile_size: Size of tiles in pixels
        overlap: Overlap between tiles in pixels
    
    Returns:
        List of (x, y) tuples for top-left corners of tiles
    """
    width, height = slide_dimensions
    stride = tile_size - overlap
    
    coords = []
    
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # Only include tiles that fit completely
            if x + tile_size <= width and y + tile_size <= height:
                coords.append((x, y))
    
    logger.info(f"Generated {len(coords)} tile coordinates (tile_size={tile_size}, overlap={overlap})")
    
    return coords


def coords_to_thumbnail(
    coords: Tuple[int, int],
    tile_size: int,
    thumbnail_shape: Tuple[int, int],
    slide_dimensions: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Map tile coordinates from slide space to thumbnail space.
    
    Args:
        coords: (x, y) coordinates in slide level 0
        tile_size: Size of tile in pixels
        thumbnail_shape: (height, width) of thumbnail
        slide_dimensions: (width, height) of slide at level 0
    
    Returns:
        Tuple of (thumb_x, thumb_y, thumb_width, thumb_height)
    """
    x, y = coords
    slide_w, slide_h = slide_dimensions
    thumb_h, thumb_w = thumbnail_shape
    
    # Scale factors
    scale_x = thumb_w / slide_w
    scale_y = thumb_h / slide_h
    
    # Map to thumbnail space
    thumb_x = int(x * scale_x)
    thumb_y = int(y * scale_y)
    thumb_tile_w = int(tile_size * scale_x)
    thumb_tile_h = int(tile_size * scale_y)
    
    # Clamp to thumbnail bounds
    thumb_x = max(0, min(thumb_x, thumb_w - 1))
    thumb_y = max(0, min(thumb_y, thumb_h - 1))
    thumb_tile_w = min(thumb_tile_w, thumb_w - thumb_x)
    thumb_tile_h = min(thumb_tile_h, thumb_h - thumb_y)
    
    return thumb_x, thumb_y, thumb_tile_w, thumb_tile_h


def extract_tile(
    slide: openslide.OpenSlide,
    coords: Tuple[int, int],
    tile_size: int,
    level: int = 0
) -> Any:
    """
    Extract a single tile from WSI.
    
    Args:
        slide: OpenSlide object
        coords: (x, y) coordinates in level 0
        tile_size: Size of tile in pixels
        level: Pyramid level to extract from
    
    Returns:
        PIL Image of extracted tile
    """
    x, y = coords
    
    # OpenSlide read_region always uses level 0 coordinates
    # but reads at the specified level
    tile = slide.read_region(
        location=(x, y),
        level=level,
        size=(tile_size, tile_size)
    )
    
    # Convert RGBA to RGB
    tile = tile.convert('RGB')
    
    return tile
