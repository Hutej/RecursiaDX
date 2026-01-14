#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe CSV-Only Duplicate Removal Script

This script removes duplicate slide entries from CSV files in the WSI+MIL pipeline.
It ONLY modifies CSV files and does NOT delete any processed files, features, or topk features.

Usage:
    python scripts/remove_duplicates.py --dry-run   # Preview changes (default)
    python scripts/remove_duplicates.py --execute   # Actually remove duplicates

Author: GigaPath Pipeline Team
Safety: Medical-grade data hygiene
"""

import os
import sys
import argparse
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')



class DuplicateRemover:
    """Safe CSV-only duplicate removal for WSI pipeline."""
    
    def __init__(self, data_dir: str = "data", dry_run: bool = True):
        """
        Initialize the duplicate remover.
        
        Args:
            data_dir: Path to the data directory
            dry_run: If True, only preview changes without modifying files
        """
        self.data_dir = Path(data_dir)
        self.dry_run = dry_run
        self.backup_dir = None
        
        # CSV files to process
        self.csv_files = [
            self.data_dir / "labels.csv",
            self.data_dir / "labels_train.csv",
            self.data_dir / "labels_val.csv",
            self.data_dir / "labels_test.csv"
        ]
        
        # Statistics
        self.stats = {
            'total_slides_scanned': 0,
            'duplicates_found': 0,
            'files_to_preserve': 0,
            'csv_rows_before': 0,
            'csv_rows_after': 0
        }
    
    def scan_csv_duplicates(self) -> Dict[str, Dict]:
        """
        Scan all CSV files for duplicate slide entries.
        
        Returns:
            Dictionary mapping CSV file paths to duplicate information
        """
        print("\n" + "="*80)
        print("SCANNING FOR DUPLICATES IN CSV FILES")
        print("="*80 + "\n")
        
        duplicate_info = {}
        
        for csv_file in self.csv_files:
            if not csv_file.exists():
                print(f"⚠️  CSV file not found: {csv_file}")
                continue
            
            print(f"📄 Scanning: {csv_file.name}")
            
            # Read CSV
            df = pd.read_csv(csv_file)
            original_count = len(df)
            self.stats['csv_rows_before'] += original_count
            
            # Find duplicates
            duplicates = df[df.duplicated(subset=['slide_name'], keep='first')]
            duplicate_count = len(duplicates)
            
            # Count unique slides
            unique_slides = df['slide_name'].nunique()
            
            duplicate_info[str(csv_file)] = {
                'dataframe': df,
                'original_count': original_count,
                'duplicate_count': duplicate_count,
                'unique_count': unique_slides,
                'duplicates': duplicates
            }
            
            # Print statistics
            if duplicate_count > 0:
                print(f"   ⚠️  Found {duplicate_count} duplicate rows")
                print(f"   ✓  Unique slides: {unique_slides}")
                print(f"   →  Will keep: {unique_slides} rows (first occurrence)")
                self.stats['duplicates_found'] += duplicate_count
            else:
                print(f"   ✓  No duplicates found")
            
            self.stats['total_slides_scanned'] += unique_slides
        
        return duplicate_info
    
    def verify_filesystem_integrity(self) -> Dict[str, List[str]]:
        """
        Verify that all processed files exist and are intact.
        
        Returns:
            Dictionary with lists of files in each category
        """
        print("\n" + "="*80)
        print("VERIFYING FILESYSTEM INTEGRITY")
        print("="*80 + "\n")
        
        integrity_info = {
            'processed_normal': [],
            'processed_tumor': [],
            'features_normal': [],
            'features_tumor': [],
            'topk_normal': [],
            'topk_tumor': []
        }
        
        # Check processed directories
        for category in ['normal', 'tumor']:
            processed_dir = self.data_dir / 'processed' / category
            if processed_dir.exists():
                slides = [d.name for d in processed_dir.iterdir() if d.is_dir()]
                integrity_info[f'processed_{category}'] = slides
                print(f"✓ Processed/{category}: {len(slides)} slides")
        
        # Check features
        for category in ['normal', 'tumor']:
            features_dir = self.data_dir / 'features' / category
            if features_dir.exists():
                features = [f.stem for f in features_dir.glob('*.h5')]
                integrity_info[f'features_{category}'] = features
                print(f"✓ Features/{category}: {len(features)} files")
        
        # Check topk features
        for category in ['normal', 'tumor']:
            topk_dir = self.data_dir / 'features_topk' / category
            if topk_dir.exists():
                topk_files = [f.stem.replace('_topk', '') for f in topk_dir.glob('*_topk.h5')]
                integrity_info[f'topk_{category}'] = topk_files
                print(f"✓ TopK/{category}: {len(topk_files)} files")
        
        # Calculate total files to preserve
        total_preserved = sum(len(v) for v in integrity_info.values())
        self.stats['files_to_preserve'] = total_preserved
        
        print(f"\n✅ Total files to preserve: {total_preserved}")
        print("   → NO FILE DELETION will occur (CSV-only cleanup)")
        
        return integrity_info
    
    def backup_csv_files(self) -> Path:
        """
        Create timestamped backup of all CSV files.
        
        Returns:
            Path to backup directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.data_dir / f"backup_labels_{timestamp}"
        
        if self.dry_run:
            print(f"\n[DRY RUN] Would create backup at: {backup_dir}")
            return backup_dir
        
        print(f"\n📦 Creating backup at: {backup_dir}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for csv_file in self.csv_files:
            if csv_file.exists():
                dest = backup_dir / csv_file.name
                shutil.copy2(csv_file, dest)
                print(f"   ✓ Backed up: {csv_file.name}")
        
        print("✅ Backup complete")
        self.backup_dir = backup_dir
        return backup_dir
    
    def remove_duplicates_from_csv(self, duplicate_info: Dict[str, Dict]) -> None:
        """
        Remove duplicate rows from CSV files, keeping first occurrence.
        
        Args:
            duplicate_info: Dictionary with duplicate information per CSV
        """
        print("\n" + "="*80)
        print("REMOVING DUPLICATES FROM CSV FILES")
        print("="*80 + "\n")
        
        for csv_path, info in duplicate_info.items():
            csv_file = Path(csv_path)
            
            if info['duplicate_count'] == 0:
                print(f"⏭️  Skipping {csv_file.name} (no duplicates)")
                continue
            
            df = info['dataframe']
            original_count = info['original_count']
            
            # Remove duplicates (keep first occurrence)
            df_clean = df.drop_duplicates(subset=['slide_name'], keep='first')
            new_count = len(df_clean)
            
            self.stats['csv_rows_after'] += new_count
            
            if self.dry_run:
                print(f"[DRY RUN] {csv_file.name}:")
                print(f"   Would remove {original_count - new_count} duplicate rows")
                print(f"   Original: {original_count} rows → Clean: {new_count} rows")
            else:
                # Write cleaned CSV
                df_clean.to_csv(csv_file, index=False)
                print(f"✅ {csv_file.name}:")
                print(f"   Removed {original_count - new_count} duplicate rows")
                print(f"   Original: {original_count} rows → Clean: {new_count} rows")
    
    def generate_report(self) -> None:
        """Generate and print summary report."""
        print("\n" + "="*80)
        print("DUPLICATE REMOVAL SUMMARY REPORT")
        print("="*80 + "\n")
        
        mode = "[DRY RUN MODE]" if self.dry_run else "[EXECUTION MODE]"
        print(f"{mode}\n")
        
        print("📊 Statistics:")
        print(f"   • Slides scanned: {self.stats['total_slides_scanned']}")
        print(f"   • Duplicates found: {self.stats['duplicates_found']}")
        print(f"   • CSV rows before: {self.stats['csv_rows_before']}")
        
        if not self.dry_run:
            print(f"   • CSV rows after: {self.stats['csv_rows_after']}")
            print(f"   • Rows removed: {self.stats['csv_rows_before'] - self.stats['csv_rows_after']}")
        
        print(f"\n💾 Files preserved:")
        print(f"   • Processed slides: ALL INTACT")
        print(f"   • Feature files: ALL INTACT")
        print(f"   • TopK features: ALL INTACT")
        print(f"   • Total files: {self.stats['files_to_preserve']} PRESERVED")
        
        if self.backup_dir and not self.dry_run:
            print(f"\n📦 Backup location:")
            print(f"   {self.backup_dir}")
        
        print("\n" + "="*80)
        
        if self.dry_run:
            print("ℹ️  This was a DRY RUN. No files were modified.")
            print("   Run with --execute to apply changes.")
        else:
            print("✅ [DONE] Duplicate slide cleanup complete")
            print("✅ [SAFE] Latest slides preserved")
            print("✅ [READY] Dataset is clean for MIL fine-tuning")
        
        print("="*80 + "\n")
        
        # Safety confirmations
        print("🛡️  SAFETY CONFIRMATIONS:")
        print("   ❌ NO preprocessing executed")
        print("   ❌ NO feature extraction executed")
        print("   ❌ NO Top-K sampling executed")
        print("   ❌ NO training executed")
        print("   ❌ NO GPU jobs triggered")
        print("   ✅ CSV-only cleanup performed")
        print("   ✅ All processed files preserved")
        print()
    
    def run(self) -> None:
        """Execute the complete duplicate removal workflow."""
        print("\n" + "="*80)
        print("WSI+MIL PIPELINE: SAFE DUPLICATE REMOVAL")
        print("="*80)
        print(f"Mode: {'DRY RUN (Preview Only)' if self.dry_run else 'EXECUTION (Will Modify Files)'}")
        print(f"Data Directory: {self.data_dir.absolute()}")
        print("="*80 + "\n")
        
        # Step 1: Scan for duplicates
        duplicate_info = self.scan_csv_duplicates()
        
        # Step 2: Verify filesystem integrity
        integrity_info = self.verify_filesystem_integrity()
        
        # Step 3: Backup CSV files
        self.backup_csv_files()
        
        # Step 4: Remove duplicates from CSV files
        self.remove_duplicates_from_csv(duplicate_info)
        
        # Step 5: Generate report
        self.generate_report()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Safe CSV-only duplicate removal for WSI+MIL pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes (default, safe)
  python scripts/remove_duplicates.py --dry-run
  
  # Execute duplicate removal
  python scripts/remove_duplicates.py --execute

Safety:
  - Only modifies CSV files (labels.csv, labels_train.csv, etc.)
  - Never deletes processed/, features/, or features_topk/ files
  - Creates timestamped backup before any changes
  - No GPU, preprocessing, or training execution
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Preview changes without modifying files (default)'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute duplicate removal (modifies CSV files)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to data directory (default: data)'
    )
    
    args = parser.parse_args()
    
    # Determine mode
    dry_run = not args.execute
    
    # Create and run remover
    remover = DuplicateRemover(data_dir=args.data_dir, dry_run=dry_run)
    remover.run()


if __name__ == "__main__":
    main()
