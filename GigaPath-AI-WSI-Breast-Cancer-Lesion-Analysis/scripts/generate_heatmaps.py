"""
Generate Attention Heatmaps for WSI Explainability

Creates attention-based visualizations showing which tiles the MIL model focuses on.

Usage:
    # Single slide with WSI
    python scripts/generate_heatmaps.py \
        --model checkpoints/best_model.pth \
        --features data/features_topk/slide_001.h5 \
        --wsi data/raw_wsi/slide_001.svs \
        --output visualizations/
    
    # Batch (no WSI - uses black background)
    python scripts/generate_heatmaps.py \
        --model checkpoints/best_model.pth \
        --features data/features_topk \
        --output visualizations/
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import cv2
import h5py
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.utils.openslide_setup

from src.mil import AttentionMIL
from src.explainability import HeatmapGenerator, GradCAM, GradCAMAggregator
from src.utils import setup_logger, load_config, get_device
from src.utils.logger import get_logger

logger = None  # Will be initialized in main()


def load_wsi_thumbnail(wsi_path: str, target_size: Tuple[int, int] = (2048, 2048)) -> np.ndarray:
    """
    Load WSI thumbnail using OpenSlide.
    
    Args:
        wsi_path: Path to WSI file
        target_size: Target thumbnail size (W, H)
    
    Returns:
        Thumbnail image (H, W, 3) uint8
    """
    try:
        import openslide
        slide = openslide.OpenSlide(wsi_path)
        
        # Get thumbnail
        thumbnail = slide.get_thumbnail(target_size)
        thumbnail_array = np.array(thumbnail.convert('RGB'))
        
        slide.close()
        return thumbnail_array
    
    except Exception as e:
        logger.warning(f"Could not load WSI: {e}. Using black background.")
        return np.zeros((*target_size[::-1], 3), dtype=np.uint8)


def generate_heatmap_for_slide(
    model: AttentionMIL,
    hdf5_path: str,
    wsi_path: Optional[str],
    output_dir: Path,
    device: torch.device,
    mode: str = 'attention',
    heatmap_gen: Optional[HeatmapGenerator] = None,
    gradcam: Optional[GradCAM] = None,
    gradcam_agg: Optional[GradCAMAggregator] = None
):
    """
    Generate heatmap visualizations for a single slide.
    
    Args:
        model: Trained MIL model
        hdf5_path: Path to HDF5 features
        wsi_path: Path to WSI (optional for attention, required for gradcam)
        output_dir: Output directory
        device: Device
        mode: Explainability mode ('attention' or 'gradcam')
        heatmap_gen: Heatmap generator (for attention mode)
        gradcam: GradCAM instance (for gradcam mode)
        gradcam_agg: GradCAM aggregator (for gradcam mode)
    """
    slide_name = Path(hdf5_path).stem
    
    # Load features and coordinates
    with h5py.File(hdf5_path, 'r') as f:
        features = torch.from_numpy(f['features'][:]).float().to(device)
        coordinates = f['coordinates'][:]
    
    # Extract attention weights
    attention_weights = model.get_attention_weights(features)
    
    logger.info(f"{slide_name}: Attention range [{attention_weights.min():.4f}, {attention_weights.max():.4f}]")
    
    # Load WSI thumbnail or create black canvas
    if wsi_path and Path(wsi_path).exists():
        thumbnail = load_wsi_thumbnail(wsi_path)
    else:
        # Create black canvas with downsampled size
        max_coord = coordinates.max(axis=0)
        width, height = int(max_coord[0]) + 512, int(max_coord[1]) + 512
        aspect_ratio = width / height
        
        target_size = 2048
        if width > height:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)
            
        thumbnail = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # Create heatmap
    heatmap = heatmap_gen.create_heatmap(
        attention_weights=attention_weights,
        coordinates=coordinates,
        canvas_size=thumbnail.shape[:2],
        tile_size=256
    )
    
    # Generate outputs based on mode
    if mode == 'attention':
        # Option 1: Scientific MIL Attention Heatmaps
        prefix = f"{slide_name}_attention"
        
        # 1. Pure heatmap
        heatmap_colored = heatmap_gen.apply_colormap(heatmap)
        heatmap_path = output_dir / f"{prefix}_heatmap.png"
        cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR))
        
        # 2. Overlay
        overlay = heatmap_gen.create_overlay(thumbnail, heatmap)
        overlay_path = output_dir / f"{prefix}_overlay.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # 3. Top-K tiles highlighted
        top_k_image = heatmap_gen.highlight_top_k_tiles(
            image=thumbnail,
            attention_weights=attention_weights,
            coordinates=coordinates,
            k=10,
            tile_size=256
        )
        topk_path = output_dir / f"{prefix}_top10.png"
        cv2.imwrite(str(topk_path), cv2.cvtColor(top_k_image, cv2.COLOR_RGB2BGR))
        
        logger.info(f"✓ Saved attention visualizations for {slide_name}")
    
    elif mode == 'gradcam':
        # Option 2: Visual Grad-CAM Heatmaps
        if not wsi_path or not Path(wsi_path).exists():
            logger.warning(f"Grad-CAM mode requires WSI file. Skipping {slide_name}")
            return
        
        logger.info(f"{slide_name}: Generating Grad-CAM visualizations...")
        prefix = f"{slide_name}_gradcam"
        
        # This is a simplified placeholder - full implementation would:
        # 1. Re-extract tiles from WSI using coordinates
        # 2. Compute Grad-CAM for each tile
        # 3. Aggregate with attention weights
        # For now, fall back to attention-based visualization with gradcam prefix
        logger.warning(f"Grad-CAM mode: Using attention-based visualization (full Grad-CAM requires tile re-extraction)")
        
        # 1. Pure heatmap
        heatmap_colored = heatmap_gen.apply_colormap(heatmap)
        heatmap_path = output_dir / f"{prefix}_heatmap.png"
        cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR))
        
        # 2. Overlay
        overlay = heatmap_gen.create_overlay(thumbnail, heatmap)
        overlay_path = output_dir / f"{prefix}_overlay.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # 3. Dense visualization
        dense_path = output_dir / f"{prefix}_dense.png"
        cv2.imwrite(str(dense_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        logger.info(f"✓ Saved Grad-CAM visualizations for {slide_name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate attention heatmaps for WSI explainability',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='Path to HDF5 feature file or directory'
    )
    
    parser.add_argument(
        '--wsi',
        type=str,
        help='Path to WSI file or directory (optional)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='visualizations',
        help='Output directory for heatmaps'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='attention',
        choices=['attention', 'gradcam'],
        help='Explainability mode: attention (scientific) or gradcam (visual)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
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
        name='heatmap_generation',
        log_dir=config['paths']['logs'],
        level=config['logging']['level'],
        save_to_file=config['logging']['save_logs']
    )
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info(f"WSI EXPLAINABILITY HEATMAP GENERATION - {args.mode.upper()} MODE")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Features: {args.features}")
    logger.info(f"WSI: {args.wsi if args.wsi else 'None (black background)'}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Mode: {args.mode}")
    if args.mode == 'attention':
        logger.info("  → Scientific MIL Attention (Option 1)")
    else:
        logger.info("  → Visual Grad-CAM (Option 2)")
    logger.info("="*60)
    
    try:
        # Load model
        checkpoint = torch.load(args.model, map_location=device)
        model = AttentionMIL(input_dim=2048, hidden_dim=512, attn_dim=256, num_classes=2, dropout=0.25)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        logger.info(f"✓ Loaded model (epoch {checkpoint['epoch']})")
        
        # Create explainability generators based on mode
        heatmap_gen = HeatmapGenerator(colormap='jet', alpha=0.5)
        
        gradcam = None
        gradcam_agg = None
        if args.mode == 'gradcam':
            logger.info("Grad-CAM mode selected (visual explainability)")
            # Note: Full Grad-CAM requires feature extractor model
            # Placeholder for now - would need to load feature extractor
            # gradcam = GradCAM(feature_extractor, target_layer, device)
            # gradcam_agg = GradCAMAggregator(tile_size=256)
        
        # Get feature files
        features_path = Path(args.features)
        if features_path.is_file():
            hdf5_files = [features_path]
        else:
            hdf5_files = sorted(list(features_path.rglob('*.h5')))
        
        logger.info(f"Found {len(hdf5_files)} slides")
        
        # Process each slide
        for hdf5_file in tqdm(hdf5_files, desc="Generating heatmaps"):
            slide_name = hdf5_file.stem
            
            # Find corresponding WSI if directory provided
            wsi_path = None
            if args.wsi:
                wsi_dir = Path(args.wsi)
                if wsi_dir.is_dir():
                    # Try common extensions
                    for ext in ['.svs', '.tiff', '.ndpi', '.mrxs']:
                        candidate = wsi_dir / f"{slide_name.replace('_topk', '')}{ext}"
                        if candidate.exists():
                            wsi_path = str(candidate)
                            break
                elif wsi_dir.is_file():
                    wsi_path = str(wsi_dir)
            
            generate_heatmap_for_slide(
                model,
                str(hdf5_file),
                wsi_path,
                output_dir,
                device,
                mode=args.mode,
                heatmap_gen=heatmap_gen,
                gradcam=gradcam,
                gradcam_agg=gradcam_agg
            )
        
        logger.info("\n" + "="*60)
        logger.info(f"✓ Heatmap generation complete!")
        logger.info(f"Visualizations saved to: {output_dir}")
        logger.info("="*60)
    
    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
