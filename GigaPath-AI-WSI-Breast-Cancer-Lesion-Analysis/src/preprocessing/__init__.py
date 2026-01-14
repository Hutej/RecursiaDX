"""
Preprocessing module for WSI tile extraction.
Supports dual modes:
- Debug mode: Save tiles as PNG for validation
- Production mode: Extract tiles transiently for direct embedding generation
"""
from .tissue_detector import TissueDetector
from .tile_extractor import TileExtractor
from .slide_utils import (
    open_slide,
    get_slide_properties,
    get_magnification,
    get_best_level_for_magnification,
    get_tile_coordinates,
    extract_tile
)

__all__ = [
    'TissueDetector',
    'TileExtractor',
    'open_slide',
    'get_slide_properties',
    'get_magnification',
    'get_best_level_for_magnification',
    'get_tile_coordinates',
    'extract_tile',
]
