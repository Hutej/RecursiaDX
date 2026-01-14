"""
CSV Deduplication System for GigaPath WSI Pipeline

Automatically removes duplicate slide entries from label CSVs with timestamped backups.

Usage:
    # Preview duplicates (dry-run)
    python scripts/deduplicate_csvs.py --dry-run
    
    # Execute deduplication
    python scripts/deduplicate_csvs.py --execute
    
    # Specify data directory
    python scripts/deduplicate_csvs.py --data-dir data/ --execute
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import shutil
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logger
from src.utils.logger import get_logger

logger = None  # Will be initialized in main()


def create_backup(csv_files: List[Path], data_dir: Path) -> Path:
    """
    Create timestamped backup of CSV files.
    
    Args:
        csv_files: List of CSV file paths to backup
        data_dir: Data directory path
    
    Returns:
        Path to backup directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = data_dir / f"backup_labels_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating backup in: {backup_dir}")
    
    for csv_file in csv_files:
        if csv_file.exists():
            backup_file = backup_dir / csv_file.name
            shutil.copy2(csv_file, backup_file)
            logger.info(f"  Backed up: {csv_file.name}")
    
    return backup_dir


def find_duplicates(df: pd.DataFrame, csv_name: str) -> Tuple[pd.DataFrame, int]:
    """
    Find and remove duplicate slide entries, keeping the latest.
    
    Args:
        df: DataFrame to deduplicate
        csv_name: Name of CSV file for logging
    
    Returns:
        Tuple of (deduplicated DataFrame, number of duplicates removed)
    """
    if 'slide_name' not in df.columns:
        logger.warning(f"{csv_name}: No 'slide_name' column found, skipping")
        return df, 0
    
    original_count = len(df)
    
    # Find duplicates
    duplicates = df.duplicated(subset=['slide_name'], keep='last')
    num_duplicates = duplicates.sum()
    
    if num_duplicates > 0:
        duplicate_names = df[duplicates]['slide_name'].unique()
        logger.info(f"{csv_name}: Found {num_duplicates} duplicate entries")
        logger.debug(f"  Duplicate slides: {', '.join(duplicate_names[:10])}")
        if len(duplicate_names) > 10:
            logger.debug(f"  ... and {len(duplicate_names) - 10} more")
        
        # Keep only last occurrence (most recent)
        df_clean = df.drop_duplicates(subset=['slide_name'], keep='last')
        
        logger.info(f"{csv_name}: Removed {num_duplicates} duplicates")
        logger.info(f"  Before: {original_count} rows → After: {len(df_clean)} rows")
    else:
        logger.info(f"{csv_name}: No duplicates found ({original_count} unique rows)")
        df_clean = df
    
    return df_clean, num_duplicates


def deduplicate_csvs(
    data_dir: str,
    dry_run: bool = True
) -> Dict[str, int]:
    """
    Deduplicate all label CSV files.
    
    Args:
        data_dir: Path to data directory
        dry_run: If True, only preview changes without saving
    
    Returns:
        Dictionary with deduplication statistics
    """
    data_path = Path(data_dir)
    
    # Define CSV files to process
    csv_files = [
        data_path / 'labels.csv',
        data_path / 'labels_train.csv',
        data_path / 'labels_val.csv',
        data_path / 'labels_test.csv'
    ]
    
    # Filter to existing files
    existing_csvs = [f for f in csv_files if f.exists()]
    
    if not existing_csvs:
        logger.warning(f"No CSV files found in {data_dir}")
        return {'total_files': 0, 'total_duplicates': 0}
    
    logger.info(f"Found {len(existing_csvs)} CSV files to process")
    
    # Statistics
    stats = {
        'total_files': len(existing_csvs),
        'total_slides': 0,
        'total_duplicates': 0,
        'files_processed': {}
    }
    
    # Dry-run mode
    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY-RUN MODE: Previewing changes (no files will be modified)")
        logger.info("=" * 60)
    
    # Create backup (only if executing)
    backup_dir = None
    if not dry_run:
        backup_dir = create_backup(existing_csvs, data_path)
    
    # Process each CSV
    for csv_file in existing_csvs:
        logger.info(f"\nProcessing: {csv_file.name}")
        
        try:
            # Load CSV
            df = pd.read_csv(csv_file)
            stats['total_slides'] += len(df)
            
            # Find and remove duplicates
            df_clean, num_duplicates = find_duplicates(df, csv_file.name)
            
            stats['total_duplicates'] += num_duplicates
            stats['files_processed'][csv_file.name] = {
                'original_rows': len(df),
                'cleaned_rows': len(df_clean),
                'duplicates_removed': num_duplicates
            }
            
            # Save cleaned CSV (only if executing)
            if not dry_run and num_duplicates > 0:
                df_clean.to_csv(csv_file, index=False)
                logger.info(f"✓ Saved cleaned {csv_file.name}")
        
        except Exception as e:
            logger.error(f"Failed to process {csv_file.name}: {e}", exc_info=True)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    if dry_run:
        logger.info("DRY-RUN SUMMARY")
    else:
        logger.info("DEDUPLICATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total CSV files: {stats['total_files']}")
    logger.info(f"Total slides scanned: {stats['total_slides']}")
    logger.info(f"Total duplicates found: {stats['total_duplicates']}")
    
    if not dry_run and backup_dir:
        logger.info(f"Backup location: {backup_dir}")
    
    logger.info("\nPer-file breakdown:")
    for filename, file_stats in stats['files_processed'].items():
        logger.info(f"  {filename}:")
        logger.info(f"    Original: {file_stats['original_rows']} rows")
        logger.info(f"    Cleaned: {file_stats['cleaned_rows']} rows")
        logger.info(f"    Removed: {file_stats['duplicates_removed']} duplicates")
    
    logger.info("=" * 60)
    
    if dry_run:
        logger.info("\n⚠️  No changes made (dry-run mode)")
        logger.info("Run with --execute to apply changes")
    else:
        logger.info("\n✅ Deduplication complete!")
        logger.info("⚠️  Note: Tile and feature files were NOT modified")
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Deduplicate slide entries in label CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to data directory containing CSV files (default: data)'
    )
    
    # Mutually exclusive group for mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    mode_group.add_argument(
        '--execute',
        action='store_true',
        help='Execute deduplication and save changes'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logger(
        name='csv_deduplication',
        log_dir='logs',
        level='INFO',
        save_to_file=True
    )
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("CSV DEDUPLICATION UTILITY")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Mode: {'DRY-RUN' if args.dry_run else 'EXECUTE'}")
    logger.info("=" * 60)
    
    # Run deduplication
    try:
        stats = deduplicate_csvs(
            data_dir=args.data_dir,
            dry_run=args.dry_run
        )
        
        # Exit code based on duplicates found
        if stats['total_duplicates'] > 0:
            if args.dry_run:
                logger.info(f"\n⚠️  Found {stats['total_duplicates']} duplicates")
                logger.info("Rerun with --execute to remove them")
                sys.exit(1)  # Non-zero to indicate action needed
            else:
                logger.info(f"\n✓ Removed {stats['total_duplicates']} duplicates")
                sys.exit(0)
        else:
            logger.info("\n✓ All CSVs are clean (no duplicates)")
            sys.exit(0)
    
    except Exception as e:
        logger.error(f"Deduplication failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
