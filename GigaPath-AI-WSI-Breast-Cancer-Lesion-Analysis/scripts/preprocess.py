"""
WSI Preprocessing CLI Script

Usage:
    # Debug mode - save tiles as PNG
    python scripts/preprocess.py --input data/raw_wsi/slide.svs --output data/processed --save-tiles

    # Production mode - extract without saving (for embedding generation)
    python scripts/preprocess.py --input data/raw_wsi --output data/processed
    
    # With config file
    python scripts/preprocess.py --config configs/config.yaml --input data/raw_wsi
"""

import argparse
import sys
import shutil
from pathlib import Path
from typing import List
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.utils.openslide_setup

from src.preprocessing import TileExtractor
from src.utils import setup_logger, load_config, set_seed
from src.utils.logger import get_logger

logger = None  # Will be initialized in main()

# Disk space guard constant
MIN_FREE_SPACE_GB = 50


def get_wsi_files(input_path: Path, extensions: List[str]) -> List[Path]:
    """
    Get list of WSI files from input path.
    
    Args:
        input_path: File or directory path
        extensions: List of file extensions to include
    
    Returns:
        List of WSI file paths
    """
    if input_path.is_file():
        if input_path.suffix.lower() in extensions:
            return [input_path]
        else:
            raise ValueError(f"File {input_path} is not a supported WSI format")
    
    elif input_path.is_dir():
        wsi_files = set()
        for ext in extensions:
            wsi_files.update(input_path.glob(f"*{ext}"))
            wsi_files.update(input_path.glob(f"*{ext.upper()}"))
        return sorted(list(wsi_files))
    
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def preprocess_slides(
    input_path: str,
    output_dir: str,
    config: dict,
    save_tiles: bool = False,
    resume: bool = False,
    slides_file: str = None
):
    """
    Preprocess WSI slides.
    
    Args:
        input_path: Path to WSI file or directory
        output_dir: Output directory for processed data
        config: Configuration dictionary
        save_tiles: If True, save tiles as PNG (debug mode)
        resume: If True, skip already processed slides
    """
    input_p = Path(input_path)
    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)
    
    # Disk space guard
    disk_usage = shutil.disk_usage(output_p)
    free_gb = disk_usage.free / (1024**3)
    if free_gb < MIN_FREE_SPACE_GB:
        logger.error(f"[DISK GUARD] Insufficient disk space: {free_gb:.1f} GB free")
        logger.error(f"[DISK GUARD] Required minimum: {MIN_FREE_SPACE_GB} GB")
        logger.error("Aborting preprocessing to prevent failure.")
        return
    logger.info(f"[DISK GUARD] Disk check passed: {free_gb:.1f} GB free")
    
    # Get WSI files
    wsi_extensions = config['preprocessing'].get('formats', ['.svs', '.tiff', '.ndpi'])
    wsi_files = get_wsi_files(input_p, wsi_extensions)
    
    if slides_file:
        slides_file_p = Path(slides_file)
        if slides_file_p.exists():
            # Handle potential BOM (UTF-16LE from PowerShell)
            try:
                raw_content = slides_file_p.read_bytes()
                if raw_content.startswith(b'\xff\xfe'):
                    content = raw_content.decode('utf-16')
                else:
                    content = raw_content.decode('utf-8', errors='ignore')
            except Exception as e:
                logger.error(f"Error reading slides file: {e}")
                return

            # support space or newline separated
            target_slides = set(content.replace('\n', ' ').split())
        
            wsi_files = [f for f in wsi_files if f.stem in target_slides]
            logger.info(f"Filtered to {len(wsi_files)} slides from {slides_file}")
            
            if len(wsi_files) == 0:
                 logger.warning(f"No matching slides found from {slides_file}")
                 return
        else:
            logger.warning(f"Slides file not found: {slides_file}")
            return

    logger.info(f"Found {len(wsi_files)} WSI files")
    
    if len(wsi_files) == 0:
        # Check for subdirectories to provide helpful guidance
        if input_p.is_dir():
            subdirs = [d.name for d in input_p.iterdir() if d.is_dir()]
            if subdirs:
                logger.info("No WSI files found at this level.")
                logger.info(f"Detected subfolders: {', '.join(subdirs)}")
                logger.info("Please run preprocessing separately for each folder.")
                return

        logger.warning("No WSI files found. Exiting.")
        return
    
    # Create tile extractor
    extractor = TileExtractor(
        tile_size=config['preprocessing']['tile_size'],
        target_magnification=config['preprocessing']['magnification'],
        overlap=config['preprocessing']['overlap'],
        tissue_threshold=config['preprocessing']['tissue_threshold'],
        save_tiles=save_tiles,
        thumbnail_size=config['preprocessing'].get('thumbnail_size', 2000)
    )
    
    # Process slides
    results = []
    failed_slides = []
    
    # Runtime disk monitoring interval (check every N slides)
    disk_check_interval = 10
    
    for idx, wsi_file in enumerate(tqdm(wsi_files, desc="Processing slides"), start=1):
        slide_name = wsi_file.stem
        
        # Runtime disk space monitoring
        if idx % disk_check_interval == 0 or idx == 1:
            disk_usage = shutil.disk_usage(output_p)
            free_gb = disk_usage.free / (1024**3)
            if free_gb < MIN_FREE_SPACE_GB:
                logger.error(f"[DISK GUARD] Runtime check failed at slide {idx}/{len(wsi_files)}")
                logger.error(f"[DISK GUARD] Free space dropped to {free_gb:.1f} GB (minimum: {MIN_FREE_SPACE_GB} GB)")
                logger.error("Aborting preprocessing to prevent failure.")
                logger.warning(f"Successfully processed {len(results)}/{len(wsi_files)} slides before abort")
                break
            else:
                logger.debug(f"[DISK GUARD] Runtime check passed: {free_gb:.1f} GB free")
        
        # Check if already processed (resume mode)
        if resume:
            # Check feature file and .done marker in features directory
            features_dir = Path(config['paths']['features'])
            feature_file = features_dir / f"{slide_name}.h5"
            done_marker = features_dir / f"{slide_name}.done"
            
            if feature_file.exists() and done_marker.exists():
                logger.info(f"[RESUME] Skipping already processed slide: {slide_name}")
                continue
        
        logger.info(f"Processing: {slide_name}")
        
        try:
            # Extract tiles
            if save_tiles:
                result = extractor.extract_tiles_from_slide(
                    str(wsi_file),
                    output_dir=str(output_p)
                )
            else:
                # In production mode, just validate without saving
                result = extractor.extract_tiles_from_slide(
                    str(wsi_file),
                    output_dir=None
                )
            
            results.append(result)
            
            logger.info(
                f"[OK] {slide_name}: {result['num_tiles_kept']} tiles "
                f"(tissue coverage: {result['tissue_coverage']:.2%})"
            )
        
        except Exception as e:
            logger.error(f"[FAIL] Failed to process {slide_name}: {e}", exc_info=True)
            failed_slides.append(slide_name)
    
    # Save summary
    summary = {
        'total_slides': len(wsi_files),
        'successful_slides': len(results),
        'failed_slides': len(failed_slides),
        'failed_slide_names': failed_slides,
        'total_tiles_extracted': sum(r['num_tiles_kept'] for r in results),
        'average_tissue_coverage': sum(r['tissue_coverage'] for r in results) / len(results) if results else 0,
        'save_tiles_mode': save_tiles
    }
    
    summary_file = output_p / 'preprocessing_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total slides: {summary['total_slides']}")
    logger.info(f"Successful: {summary['successful_slides']}")
    logger.info(f"Failed: {summary['failed_slides']}")
    logger.info(f"Total tiles: {summary['total_tiles_extracted']}")
    logger.info(f"Avg tissue coverage: {summary['average_tissue_coverage']:.2%}")
    logger.info(f"Mode: {'DEBUG (tiles saved)' if save_tiles else 'PRODUCTION (transient)'}")
    logger.info("="*60)
    
    if failed_slides:
        logger.warning(f"Failed slides: {', '.join(failed_slides)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Preprocess WSI slides for tile extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input WSI file or directory'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    
    parser.add_argument(
        '--save-tiles',
        action='store_true',
        help='Save tiles as PNG (debug mode). Default: production mode (transient)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip already processed slides'
    )

    parser.add_argument(
        '--slides_file',
        type=str,
        help='Path to file containing list of slide names to process'
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Setup logging
    global logger
    logger = setup_logger(
        name='preprocessing',
        log_dir=config['paths']['logs'],
        level=config['logging']['level'],
        save_to_file=config['logging']['save_logs']
    )
    
    # Set random seed for reproducibility
    set_seed(
        seed=config['experiment']['seed'],
        deterministic=config['experiment']['deterministic']
    )
    
    # Log configuration
    logger.info("="*60)
    logger.info("WSI PREPROCESSING")
    logger.info("="*60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Tile size: {config['preprocessing']['tile_size']}")
    logger.info(f"Magnification: {config['preprocessing']['magnification']}x")
    logger.info(f"Tissue threshold: {config['preprocessing']['tissue_threshold']}")
    logger.info(f"Mode: {'DEBUG (save tiles)' if args.save_tiles else 'PRODUCTION (transient)'}")
    logger.info("="*60)
    
    # Run preprocessing
    try:
        preprocess_slides(
            input_path=args.input,
            output_dir=args.output,
            config=config,
            save_tiles=args.save_tiles,
            resume=args.resume,
            slides_file=args.slides_file
        )
        
        logger.info("Preprocessing complete!")
    
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
