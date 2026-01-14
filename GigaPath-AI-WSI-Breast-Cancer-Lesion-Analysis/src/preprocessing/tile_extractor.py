"""
Tile extraction module with dual-mode support:
1. Debug mode: Save tiles as PNG for validation
2. Production mode: Extract tiles transiently for direct embedding generation
"""
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Generator
import csv

from .tissue_detector import TissueDetector
from .slide_utils import (
    open_slide,
    get_slide_properties,
    get_best_level_for_magnification,
    get_tile_coordinates,
    extract_tile
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TileExtractor:
    """Extract tiles from WSI with tissue filtering."""
    
    def __init__(
        self,
        tile_size: int = 256,
        target_magnification: float = 20.0,
        overlap: int = 0,
        tissue_threshold: float = 0.5,
        save_tiles: bool = False,  # NEW: Control tile saving
        thumbnail_size: int = 2000
    ):
        """
        Initialize tile extractor.
        
        Args:
            tile_size: Size of tiles in pixels
            target_magnification: Target magnification level
            overlap: Overlap between tiles in pixels
            tissue_threshold: Minimum tissue ratio to keep tile
            save_tiles: If True, save tiles to disk (debug mode)
            thumbnail_size: Size for tissue detection thumbnail
        """
        self.tile_size = tile_size
        self.target_magnification = target_magnification
        self.overlap = overlap
        self.tissue_threshold = tissue_threshold
        self.save_tiles = save_tiles
        
        self.tissue_detector = TissueDetector(thumbnail_size=thumbnail_size)
    
    def extract_tiles_from_slide(
        self,
        slide_path: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract tiles from a single WSI.
        
        Args:
            slide_path: Path to WSI file
            output_dir: Output directory (required if save_tiles=True)
        
        Returns:
            Dictionary with extraction statistics and tile information
        """
        if self.save_tiles and output_dir is None:
            raise ValueError("output_dir required when save_tiles=True")
        
        # Open slide
        slide = open_slide(slide_path)
        slide_name = Path(slide_path).stem
        
        logger.info(f"Processing slide: {slide_name}")
        
        # Get slide properties
        properties = get_slide_properties(slide)
        logger.info(f"Slide dimensions: {properties['dimensions']}")
        
        # Detect tissue
        tissue_mask, tissue_coverage = self.tissue_detector.detect_tissue(slide)
        logger.info(f"Tissue coverage: {tissue_coverage:.2%}")
        
        # Find best level for target magnification
        level, actual_mag = get_best_level_for_magnification(
            slide,
            self.target_magnification
        )
        
        # Generate tile coordinates
        coords_list = get_tile_coordinates(
            properties['dimensions'],
            self.tile_size,
            self.overlap
        )
        
        # Filter by tissue
        filtered_coords = self._filter_by_tissue(
            coords_list,
            tissue_mask,
            properties['dimensions']
        )
        
        logger.info(f"Tiles after tissue filtering: {len(filtered_coords)} / {len(coords_list)}")
        
        # Setup output if saving tiles
        if self.save_tiles:
            slide_output_dir = Path(output_dir) / slide_name
            tiles_dir = slide_output_dir / 'tiles'
            tiles_dir.mkdir(parents=True, exist_ok=True)
        else:
            slide_output_dir = None
            tiles_dir = None
        
        # Extract tiles
        tile_metadata = []
        
        for idx, (coords, tissue_ratio) in enumerate(filtered_coords):
            # Extract tile
            tile = extract_tile(slide, coords, self.tile_size, level)
            
            # Create metadata
            tile_info = {
                'tile_id': idx,
                'x': coords[0],
                'y': coords[1],
                'level': level,
                'tissue_ratio': float(tissue_ratio),
                'magnification': actual_mag
            }
            
            # Save tile if in debug mode
            if self.save_tiles:
                tile_filename = f"tile_{coords[0]}_{coords[1]}.png"
                tile_path = tiles_dir / tile_filename
                tile.save(tile_path)
                tile_info['filename'] = tile_filename
            
            tile_metadata.append(tile_info)
        
        # Save metadata
        result = {
            'slide_name': slide_name,
            'slide_path': slide_path,
            'properties': properties,
            'tissue_coverage': tissue_coverage,
            'num_tiles_total': len(coords_list),
            'num_tiles_kept': len(filtered_coords),
            'level': level,
            'magnification': actual_mag,
            'tiles': tile_metadata
        }
        
        if self.save_tiles:
            self._save_metadata(result, slide_output_dir)
        
        # Close slide
        slide.close()
        
        return result
    
    def extract_tiles_generator(
        self,
        slide_path: str
    ) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
        """
        Extract tiles as a generator for on-the-fly processing.
        
        This is the PRODUCTION MODE for direct embedding generation.
        Tiles are extracted transiently and not saved to disk.
        
        Args:
            slide_path: Path to WSI file
        
        Yields:
            Tuple of (tile_array, tile_metadata)
            - tile_array: numpy array (256, 256, 3)
            - tile_metadata: dict with coords, tissue_ratio, etc.
        """
        # Open slide
        slide = open_slide(slide_path)
        slide_name = Path(slide_path).stem
        
        logger.info(f"Extracting tiles (generator mode): {slide_name}")
        
        # Get slide properties
        properties = get_slide_properties(slide)
        
        # Detect tissue
        tissue_mask, tissue_coverage = self.tissue_detector.detect_tissue(slide)
        
        # Find best level
        level, actual_mag = get_best_level_for_magnification(
            slide,
            self.target_magnification
        )
        
        # Generate and filter coordinates
        coords_list = get_tile_coordinates(
            properties['dimensions'],
            self.tile_size,
            self.overlap
        )
        
        filtered_coords = self._filter_by_tissue(
            coords_list,
            tissue_mask,
            properties['dimensions']
        )
        
        logger.info(f"Yielding {len(filtered_coords)} tiles...")
        
        # Extract and yield tiles
        for idx, (coords, tissue_ratio) in enumerate(filtered_coords):
            # Extract tile
            tile = extract_tile(slide, coords, self.tile_size, level)
            
            # Convert to numpy array
            tile_array = np.array(tile)
            
            # Create metadata
            tile_metadata = {
                'slide_name': slide_name,
                'tile_id': idx,
                'x': coords[0],
                'y': coords[1],
                'level': level,
                'tissue_ratio': float(tissue_ratio),
                'magnification': actual_mag
            }
            
            yield tile_array, tile_metadata
        
        # Close slide
        slide.close()
        
        logger.info(f"Tile extraction complete: {slide_name}")
    
    def _filter_by_tissue(
        self,
        coords_list: List[Tuple[int, int]],
        tissue_mask: np.ndarray,
        slide_dimensions: Tuple[int, int]
    ) -> List[Tuple[Tuple[int, int], float]]:
        """
        Filter tile coordinates by tissue ratio.
        
        Args:
            coords_list: List of (x, y) coordinates
            tissue_mask: Binary tissue mask
            slide_dimensions: (width, height) of slide
        
        Returns:
            List of ((x, y), tissue_ratio) tuples for tiles passing threshold
        """
        filtered = []
        
        for coords in coords_list:
            tissue_ratio = self.tissue_detector.calculate_tissue_ratio(
                coords,
                self.tile_size,
                tissue_mask,
                slide_dimensions
            )
            
            if tissue_ratio >= self.tissue_threshold:
                filtered.append((coords, tissue_ratio))
        
        return filtered
    
    def _save_metadata(
        self,
        result: Dict[str, Any],
        output_dir: Path
    ):
        """
        Save tile metadata to files.
        
        Args:
            result: Extraction result dictionary
            output_dir: Output directory
        """
        # Save coordinates as CSV
        coords_file = output_dir / 'coordinates.csv'
        with open(coords_file, 'w', newline='') as f:
            if len(result['tiles']) > 0:
                fieldnames = ['tile_id', 'x', 'y', 'level', 'tissue_ratio', 'magnification']
                if 'filename' in result['tiles'][0]:
                    fieldnames.append('filename')
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(result['tiles'])
        
        logger.info(f"Coordinates saved to {coords_file}")
        
        # Save metadata as JSON
        metadata_file = output_dir / 'metadata.json'
        
        # Create clean metadata (without tile list for brevity)
        clean_metadata = {
            'slide_name': result['slide_name'],
            'slide_path': result['slide_path'],
            'dimensions': result['properties']['dimensions'],
            'level_count': result['properties']['level_count'],
            'magnification': result['magnification'],
            'tissue_coverage': result['tissue_coverage'],
            'num_tiles_total': result['num_tiles_total'],
            'num_tiles_kept': result['num_tiles_kept'],
            'level_used': result['level']
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(clean_metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_file}")
