"""
Top-K Tile Sampling CLI Script

Selects the K most informative tiles from HDF5 feature files to reduce
computational cost for MIL training.

Usage:
    # Sample all feature files
    python scripts/sample_tiles.py --input data/features --output data/features_topk
    
    # Single file
    python scripts/sample_tiles.py --input data/features/slide_001.h5 --output data/features_topk
    
    # Custom K value
    python scripts/sample_tiles.py --input data/features --output data/features_topk --k 500
    
    # Resume
    python scripts/sample_tiles.py --input data/features --output data/features_topk --resume
"""
import argparse
import sys
from pathlib import Path
from typing import List
from tqdm import tqdm
import torch
import json

# === GPU ENFORCEMENT ===
if not torch.cuda.is_available():
    raise RuntimeError("CUDA NOT AVAILABLE — GPU REQUIRED FOR TOP-K SAMPLING")

# Strict check: no CPU fallback allowed
if torch.device("cuda").type != "cuda":
    raise RuntimeError("CUDA DEVICE ERROR")

gpu_name = torch.cuda.get_device_name(0)
print(f"[GPU ONLY] Top-K sampling locked to CUDA: {gpu_name}")
# =======================

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sampling import TopKSelector
from src.mil import AttentionMIL
from src.utils import setup_logger, load_config, set_seed
from src.utils.logger import get_logger

logger = None  # Will be initialized in main()


def load_mil_model(checkpoint_path: str, device: torch.device) -> AttentionMIL:
    """
    Load trained MIL model for attention-based ranking.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: CUDA device
    
    Returns:
        Loaded MIL model in eval mode
    """
    logger.info(f"Loading MIL model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with same architecture
    model = AttentionMIL(
        input_dim=2048,
        hidden_dim=512,
        attn_dim=256,
        num_classes=2,
        dropout=0.25
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"✓ Model loaded (epoch: {checkpoint['epoch']})")
    return model


def get_hdf5_files(input_path: Path) -> List[Path]:
    """Get list of HDF5 files from input path."""
    if input_path.is_file():
        if input_path.suffix.lower() in ['.h5', '.hdf5']:
            return [input_path]
        else:
            raise ValueError(f"File {input_path} is not an HDF5 file")
    
    elif input_path.is_dir():
        h5_files = list(input_path.glob('*.h5')) + list(input_path.glob('*.hdf5'))
        return sorted(h5_files)
    
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def sample_top_k_from_files(
    input_path: str,
    output_dir: str,
    k: int,
    ranking_method: str,
    alpha: float = 0.7,
    mil_model: torch.nn.Module = None,
    device: torch.device = None,
    resume: bool = False,
    normalize: bool = True
):
    """
    Apply Top-K sampling to HDF5 feature files.
    
    Args:
        input_path: Path to HDF5 file or directory
        output_dir: Output directory for sampled HDF5 files
        k: Number of top tiles to select
        ranking_method: 'feature_norm', 'attention', or 'weighted'
        alpha: Weight for attention in weighted method
        mil_model: Trained MIL model (for attention/weighted)
        device: CUDA device (for attention/weighted)
        resume: If True, skip already processed files
        normalize: If True, L2-normalize features before ranking (default: True)
    """
    input_p = Path(input_path)
    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)
    
    # Get HDF5 files
    hdf5_files = get_hdf5_files(input_p)
    
    logger.info(f"Found {len(hdf5_files)} HDF5 files")
    
    if len(hdf5_files) == 0:
        logger.warning("No HDF5 files found. Exiting.")
        return
    
    # Create Top-K selector
    selector = TopKSelector(
        k=k,
        ranking_method=ranking_method,
        alpha=alpha,
        mil_model=mil_model,
        device=device,
        normalize_features=normalize
    )
    
    # Process files
    results = []
    failed_files = []
    total_tiles_before = 0
    total_tiles_after = 0
    
    for hdf5_file in tqdm(hdf5_files, desc="Sampling tiles"):
        file_name = hdf5_file.stem
        
        # Determine output filename
        if file_name.endswith('_topk'):
            output_name = file_name
        else:
            output_name = f"{file_name}_topk"
        
        output_file = output_p / f"{output_name}.h5"
        
        # Check if already processed (resume mode)
        if resume and output_file.exists():
            logger.info(f"Skipping {file_name} (already processed)")
            continue
        
        logger.info(f"Processing: {file_name}")
        
        try:
            # Apply Top-K sampling
            stats = selector.select_top_k_from_hdf5(
                str(hdf5_file),
                str(output_file)
            )
            
            results.append(stats)
            total_tiles_before += stats['original_num_tiles']
            total_tiles_after += stats['selected_k']
            
            logger.info(
                f"✓ {file_name}: {stats['original_num_tiles']} → {stats['selected_k']} tiles "
                f"({stats['reduction_ratio']:.1%} reduction)"
            )
        
        except Exception as e:
            logger.error(f"✗ Failed to process {file_name}: {e}", exc_info=True)
            failed_files.append(file_name)
    
    # Calculate overall statistics
    overall_reduction = 1 - (total_tiles_after / total_tiles_before) if total_tiles_before > 0 else 0
    
    # Save summary
    summary = {
        'total_files': len(hdf5_files),
        'successful_files': len(results),
        'failed_files': len(failed_files),
        'failed_file_names': failed_files,
        'k_value': k,
        'ranking_method': ranking_method,
        'total_tiles_before': total_tiles_before,
        'total_tiles_after': total_tiles_after,
        'overall_reduction_ratio': overall_reduction,
        'average_tiles_per_slide_before': total_tiles_before / len(results) if results else 0,
        'average_tiles_per_slide_after': total_tiles_after / len(results) if results else 0
    }
    
    summary_file = output_p / 'top_k_sampling_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TOP-K SAMPLING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total files: {summary['total_files']}")
    logger.info(f"Successful: {summary['successful_files']}")
    logger.info(f"Failed: {summary['failed_files']}")
    logger.info(f"K value: {summary['k_value']}")
    logger.info(f"Ranking method: {summary['ranking_method']}")
    logger.info(f"Total tiles before: {summary['total_tiles_before']}")
    logger.info(f"Total tiles after: {summary['total_tiles_after']}")
    logger.info(f"Overall reduction: {summary['overall_reduction_ratio']:.1%}")
    logger.info(f"Avg tiles/slide: {summary['average_tiles_per_slide_before']:.0f} → "
                f"{summary['average_tiles_per_slide_after']:.0f}")
    logger.info("="*60)
    
    if failed_files:
        logger.warning(f"Failed files: {', '.join(failed_files)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Apply Top-K tile sampling to HDF5 feature files',
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
        help='Input HDF5 file or directory'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/features_topk',
        help='Output directory for sampled HDF5 files'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        help='Number of top tiles to select (overrides config)'
    )
    
    parser.add_argument(
        '--ranking-method',
        type=str,
        choices=['feature_norm', 'attention', 'weighted'],
        help='Ranking method (overrides config)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to trained MIL model (required for attention/weighted methods)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.7,
        help='Weight for attention in weighted method (default: 0.7)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip already processed files'
    )
    
    parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='L2-normalize features before ranking (recommended for reproducibility, default: True)'
    )
    
    parser.add_argument(
        '--no-normalize',
        dest='normalize',
        action='store_false',
        help='Disable feature normalization (legacy mode)'
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Get parameters from config or args
    k = args.k if args.k is not None else config['sampling']['k']
    ranking_method = args.ranking_method if args.ranking_method else config['sampling']['ranking_method']
    
    # Setup logging
    global logger
    logger = setup_logger(
        name='top_k_sampling',
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
    logger.info("TOP-K TILE SAMPLING")
    logger.info("="*60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"K value: {k}")
    logger.info(f"Ranking method: {ranking_method}")
    if ranking_method == 'weighted':
        logger.info(f"Alpha (attention weight): {args.alpha}")
    if ranking_method in ['attention', 'weighted']:
        logger.info(f"MIL model: {args.model}")
    logger.info(f"Resume: {args.resume}")
    logger.info("="*60)
    
    # Load MIL model if needed
    mil_model = None
    device = None
    if ranking_method in ['attention', 'weighted']:
        if not args.model:
            logger.error(f"{ranking_method} method requires --model argument")
            sys.exit(1)
        
        device = torch.device('cuda')
        mil_model = load_mil_model(args.model, device)
    
    # Run Top-K sampling
    try:
        sample_top_k_from_files(
            input_path=args.input,
            output_dir=args.output,
            k=k,
            ranking_method=ranking_method,
            alpha=args.alpha,
            mil_model=mil_model,
            device=device,
            resume=args.resume,
            normalize=args.normalize
        )
        
        logger.info("Top-K sampling complete!")
    
    except Exception as e:
        logger.error(f"Top-K sampling failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
