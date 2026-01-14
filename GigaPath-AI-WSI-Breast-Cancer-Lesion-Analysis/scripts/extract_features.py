"""
Feature Extraction CLI Script

Extracts deep learning features from WSI tiles using frozen backbone.
Features are cached to HDF5 for efficient MIL training.

Usage:
    # Extract features for all WSI
    python scripts/extract_features.py --input data/raw_wsi --output data/features
    
    # Single slide
    python scripts/extract_features.py --input data/raw_wsi/slide.svs --output data/features
    
    # With config
    python scripts/extract_features.py --config configs/config.yaml
    
    # Resume after interruption
    python scripts/extract_features.py --input data/raw_wsi --output data/features --resume
"""

import argparse
import sys
import shutil
from pathlib import Path
from typing import List
from tqdm import tqdm
import json
import time
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.utils.openslide_setup

from src.feature_extraction import FeatureExtractor
from src.preprocessing import TileExtractor
from src.utils import setup_logger, load_config, set_seed, GPUMonitor
from src.utils.logger import get_logger

logger = None  # Will be initialized in main()


def get_wsi_files(input_path: Path, extensions: List[str]) -> List[Path]:
    """Get list of WSI files from input path."""
    if input_path.is_file():
        if input_path.suffix.lower() in extensions:
            return [input_path]
        else:
            raise ValueError(f"File {input_path} is not a supported WSI format")
    
    elif input_path.is_dir():
        wsi_files = []
        for ext in extensions:
            wsi_files.extend(input_path.glob(f"*{ext}"))
            wsi_files.extend(input_path.glob(f"*{ext.upper()}"))
        return sorted(wsi_files)
    
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def is_preprocessed_slide_dir(path: Path) -> bool:
    """
    Check if path is a preprocessed slide directory.
    
    Args:
        path: Path to check
    
    Returns:
        True if path contains tiles/ subdirectory and coordinates.csv
    """
    tiles_dir = path / "tiles"
    coords_file = path / "coordinates.csv"
    return tiles_dir.exists() and tiles_dir.is_dir() and coords_file.exists()



def extract_features_from_slides(
    input_path: str,
    output_dir: str,
    config: dict,
    resume: bool = False
):
    """
    Extract features from WSI slides.
    
    Args:
        input_path: Path to WSI file or directory
        output_dir: Output directory for HDF5 features
        config: Configuration dictionary
        resume: If True, skip already processed slides
    """
    input_p = Path(input_path)
    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)
    
    # Get WSI files
    wsi_extensions = config['preprocessing'].get('formats', ['.svs', '.tiff', '.ndpi'])
    wsi_files = get_wsi_files(input_p, wsi_extensions)
    
    logger.info(f"Found {len(wsi_files)} WSI files")
    
    # If no WSI files, check for preprocessed slide directories
    if len(wsi_files) == 0:
        if input_p.is_dir():
            # Check for preprocessed slide directories
            slide_dirs = [d for d in input_p.iterdir() if d.is_dir() and is_preprocessed_slide_dir(d)]
            
            if len(slide_dirs) > 0:
                logger.info(f"Detected {len(slide_dirs)} preprocessed slide directories")
                logger.info("Processing preprocessed tiles...")
                
                # Setup GPU monitor
                gpu_monitor = GPUMonitor() if config['hardware']['gpu_id'] >= 0 else None
                if gpu_monitor:
                    gpu_monitor.log_memory_status()
                
                # Create feature extractor
                feature_extractor = FeatureExtractor(
                    model_name=config['feature_extraction']['backbone'],
                    batch_size=config['feature_extraction']['batch_size'],
                    use_amp=config['hardware']['mixed_precision']
                )
                
                # Process slides
                results = []
                failed_slides = []
                total_tiles = 0
                start_time = time.time()
                
                for slide_dir in tqdm(slide_dirs, desc="Extracting features"):
                    slide_name = slide_dir.name
                    output_file = output_p / f"{slide_name}.h5"
                    
                    # Check if already processed (resume mode)
                    if resume:
                        done_marker = output_p / f"{slide_name}.done"
                        if output_file.exists() and done_marker.exists():
                            logger.info(f"[RESUME] Skipping {slide_name} (already processed)")
                            continue
                    
                    logger.info(f"Processing: {slide_name}")
                    
                    try:
                        slide_start = time.time()
                        
                        # Extract features from preprocessed tiles
                        stats = feature_extractor.extract_features_from_tiles(
                            tiles_dir=slide_dir / "tiles",
                            coordinates_csv=slide_dir / "coordinates.csv",
                            output_path=output_file,
                            slide_name=slide_name
                        )
                        
                        slide_time = time.time() - slide_start
                        stats['processing_time_seconds'] = slide_time
                        
                        results.append(stats)
                        total_tiles += stats['num_tiles']
                        
                        logger.info(
                            f"[OK] {slide_name}: {stats['num_tiles']} tiles, "
                            f"{stats['feature_dim']}-dim features, "
                            f"{slide_time:.1f}s"
                        )
                        
                        # Create .done marker after successful feature extraction
                        done_marker = output_p / f"{slide_name}.done"
                        done_marker.touch()
                        logger.info(f"[RESUME] Completed slide: {slide_name}")
                        
                        # Auto tile cleanup
                        tiles_dir = slide_dir / "tiles"
                        if tiles_dir.exists():
                            try:
                                shutil.rmtree(tiles_dir, ignore_errors=True)
                                logger.info(f"[CLEANUP] Deleted tiles for slide: {slide_name}")
                            except Exception as e:
                                logger.warning(f"[CLEANUP] Could not delete tiles for {slide_name}: {e}")
                        
                        # Log GPU memory if available
                        if gpu_monitor:
                            gpu_monitor.log_memory_status()
                    
                    except Exception as e:
                        logger.error(f"[FAIL] Failed to process {slide_name}: {e}", exc_info=True)
                        failed_slides.append(slide_name)
                
                total_time = time.time() - start_time
                
                # Summary
                logger.info("="*60)
                logger.info("Feature Extraction Summary")
                logger.info("="*60)
                logger.info(f"Processed: {len(results)}/{len(slide_dirs)} slides")
                logger.info(f"Total tiles: {total_tiles}")
                logger.info(f"Total time: {total_time:.1f}s")
                logger.info(f"Average time per slide: {total_time/len(results):.1f}s" if results else "N/A")
                
                if failed_slides:
                    logger.warning(f"Failed slides: {', '.join(failed_slides)}")
                
                return
            
            # Fallback: no preprocessed dirs, just subdirectories
            subdirs = [d.name for d in input_p.iterdir() if d.is_dir()]
            if subdirs:
                logger.info("No WSI files found at this level.")
                logger.info(f"Detected subfolders: {', '.join(subdirs)}")
                logger.info("Please run feature extraction separately for each folder.")
                return

        logger.warning("No WSI files found. Exiting.")
        return
    
    # Setup GPU monitor
    gpu_monitor = GPUMonitor() if config['hardware']['gpu_id'] >= 0 else None
    if gpu_monitor:
        gpu_monitor.log_memory_status()
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(
        model_name=config['feature_extraction']['backbone'],
        batch_size=config['feature_extraction']['batch_size'],
        use_amp=config['hardware']['mixed_precision']
    )
    
    # Create tile extractor (production mode - no tile saving)
    tile_extractor = TileExtractor(
        tile_size=config['preprocessing']['tile_size'],
        target_magnification=config['preprocessing']['magnification'],
        overlap=config['preprocessing']['overlap'],
        tissue_threshold=config['preprocessing']['tissue_threshold'],
        save_tiles=False,  # Transient extraction for production
    )
    
    # Process slides
    results = []
    failed_slides = []
    total_tiles = 0
    start_time = time.time()
    
    for wsi_file in tqdm(wsi_files, desc="Extracting features"):
        slide_name = wsi_file.stem
        output_file = output_p / f"{slide_name}.h5"
        
        # Check if already processed (resume mode)
        if resume:
            done_marker = output_p / f"{slide_name}.done"
            if output_file.exists() and done_marker.exists():
                logger.info(f"[RESUME] Skipping {slide_name} (already processed)")
                continue
        
        logger.info(f"Processing: {slide_name}")
        
        try:
            slide_start = time.time()
            
            # Extract features
            stats = feature_extractor.extract_features_from_slide(
                slide_path=str(wsi_file),
                output_path=str(output_file),
                tile_extractor=tile_extractor
            )
            
            slide_time = time.time() - slide_start
            stats['processing_time_seconds'] = slide_time
            
            results.append(stats)
            total_tiles += stats['num_tiles']
            
            logger.info(
                f"[OK] {slide_name}: {stats['num_tiles']} tiles, "
                f"{stats['feature_dim']}-dim features, "
                f"{slide_time:.1f}s"
            )
            
            # Create .done marker after successful feature extraction  
            done_marker = output_p / f"{slide_name}.done"
            done_marker.touch()
            logger.info(f"[RESUME] Completed slide: {slide_name}")
            
            # Auto tile cleanup (for saved tiles from preprocessing)
            # Note: In production mode tiles are not saved, but check anyway
            tile_dir = Path(config['paths']['processed_tiles']) / slide_name / "tiles"
            if tile_dir.exists():
                try:
                    shutil.rmtree(tile_dir, ignore_errors=True)
                    logger.info(f"[CLEANUP] Deleted tiles for slide: {slide_name}")
                except Exception as e:
                    logger.warning(f"[CLEANUP] Could not delete tiles for {slide_name}: {e}")
            
            # Log GPU memory if available
            if gpu_monitor:
                gpu_monitor.log_memory_status()
        
        except Exception as e:
            logger.error(f"[FAIL] Failed to process {slide_name}: {e}", exc_info=True)
            failed_slides.append(slide_name)
    
    total_time = time.time() - start_time
    
    # Save summary
    summary = {
        'total_slides': len(wsi_files),
        'successful_slides': len(results),
        'failed_slides': len(failed_slides),
        'failed_slide_names': failed_slides,
        'total_tiles_processed': total_tiles,
        'average_tiles_per_slide': total_tiles / len(results) if results else 0,
        'model_name': config['feature_extraction']['backbone'],
        'feature_dim': results[0]['feature_dim'] if results else 0,
        'batch_size': config['feature_extraction']['batch_size'],
        'total_time_seconds': total_time,
        'average_time_per_slide_seconds': total_time / len(results) if results else 0
    }
    
    summary_file = output_p / 'feature_extraction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("FEATURE EXTRACTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total slides: {summary['total_slides']}")
    logger.info(f"Successful: {summary['successful_slides']}")
    logger.info(f"Failed: {summary['failed_slides']}")
    logger.info(f"Total tiles: {summary['total_tiles_processed']}")
    logger.info(f"Avg tiles/slide: {summary['average_tiles_per_slide']:.0f}")
    logger.info(f"Model: {summary['model_name']}")
    logger.info(f"Feature dim: {summary['feature_dim']}")
    logger.info(f"Batch size: {summary['batch_size']}")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Avg time/slide: {summary['average_time_per_slide_seconds']/60:.1f} minutes")
    logger.info("="*60)
    
    if failed_slides:
        logger.warning(f"Failed slides: {', '.join(failed_slides)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract features from WSI slides using frozen backbone',
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
        default='data/features',
        help='Output directory for HDF5 feature files'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip already processed slides'
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
        name='feature_extraction',
        log_dir=config['paths']['logs'],
        level=config['logging']['level'],
        save_to_file=config['logging']['save_logs']
    )
    
    # Set random seed for reproducibility
    set_seed(
        seed=config['experiment']['seed'],
        deterministic=config['experiment']['deterministic']
    )
    
    # ------------------------------------------------------------------------
    # STEP 3: FORCE GPU-ONLY MODE (NO CPU FALLBACK)
    # ------------------------------------------------------------------------
    if not torch.cuda.is_available():
        logger.error("GPU REQUIRED — CPU EXECUTION DISALLOWED")
        raise RuntimeError("CUDA NOT AVAILABLE — GPU REQUIRED")

    # Force device assignment
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Disable CPU fallback globally
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    logger.info("GPU ENFORCEMENT ACTIVE")
    logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    # ------------------------------------------------------------------------
    
    # Log configuration
    logger.info("="*60)
    logger.info("FEATURE EXTRACTION")
    logger.info("="*60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Model: {config['feature_extraction']['backbone']}")
    logger.info(f"Batch size: {config['feature_extraction']['batch_size']}")
    logger.info(f"Tile size: {config['preprocessing']['tile_size']}")
    logger.info(f"Magnification: {config['preprocessing']['magnification']}×")
    logger.info(f"Mixed precision: {config['hardware']['mixed_precision']}")
    logger.info(f"Resume: {args.resume}")
    logger.info("="*60)
    
    # Run feature extraction
    try:
        extract_features_from_slides(
            input_path=args.input,
            output_dir=args.output,
            config=config,
            resume=args.resume
        )
        
        logger.info("Feature extraction complete!")
    
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
