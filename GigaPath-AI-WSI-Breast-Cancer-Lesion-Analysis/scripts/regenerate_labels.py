#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate Labels CSV Files from features_topk Directory

This script scans the features_topk/ directory and regenerates all label CSV files
with proper train/val/test stratified splits.

Usage:
    python scripts/regenerate_labels.py

Author: GigaPath Pipeline Team
"""

import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def scan_features_topk(data_dir: Path) -> pd.DataFrame:
    """
    Scan features_topk directory and create labels DataFrame.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        DataFrame with slide_name and label columns
    """
    print("="*80)
    print("SCANNING FEATURES_TOPK DIRECTORY")
    print("="*80 + "\n")
    
    slides = []
    
    # Scan normal slides
    normal_dir = data_dir / 'features_topk' / 'normal'
    if normal_dir.exists():
        normal_files = list(normal_dir.glob('*_topk.h5'))
        for f in normal_files:
            slide_id = f.stem.replace('_topk', '')
            slides.append({
                'slide_name': f'normal/{slide_id}',
                'label': 0
            })
        print(f"✓ Found {len(normal_files)} normal slides")
    else:
        print(f"⚠ Warning: {normal_dir} not found")
    
    # Scan tumor slides
    tumor_dir = data_dir / 'features_topk' / 'tumor'
    if tumor_dir.exists():
        tumor_files = list(tumor_dir.glob('*_topk.h5'))
        for f in tumor_files:
            slide_id = f.stem.replace('_topk', '')
            slides.append({
                'slide_name': f'tumor/{slide_id}',
                'label': 1
            })
        print(f"✓ Found {len(tumor_files)} tumor slides")
    else:
        print(f"⚠ Warning: {tumor_dir} not found")
    
    # Create DataFrame
    df = pd.DataFrame(slides)
    df = df.sort_values('slide_name').reset_index(drop=True)
    
    print(f"\n📊 Total slides found: {len(df)}")
    print(f"   • Normal (label=0): {(df['label'] == 0).sum()}")
    print(f"   • Tumor (label=1): {(df['label'] == 1).sum()}")
    
    return df


def create_stratified_splits(df: pd.DataFrame, 
                            train_size: float = 0.7,
                            val_size: float = 0.15,
                            test_size: float = 0.15,
                            random_state: int = 42) -> tuple:
    """
    Create stratified train/val/test splits.
    
    Args:
        df: Full DataFrame with all slides
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("\n" + "="*80)
    print("CREATING STRATIFIED SPLITS")
    print("="*80 + "\n")
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
    
    # Second split: separate train and val
    val_proportion = val_size / (train_size + val_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_proportion,
        stratify=train_val_df['label'],
        random_state=random_state
    )
    
    # Print statistics
    print(f"Split proportions: Train={train_size:.0%}, Val={val_size:.0%}, Test={test_size:.0%}\n")
    
    print(f"📊 Training Set: {len(train_df)} slides")
    print(f"   • Normal: {(train_df['label'] == 0).sum()}")
    print(f"   • Tumor: {(train_df['label'] == 1).sum()}")
    
    print(f"\n📊 Validation Set: {len(val_df)} slides")
    print(f"   • Normal: {(val_df['label'] == 0).sum()}")
    print(f"   • Tumor: {(val_df['label'] == 1).sum()}")
    
    print(f"\n📊 Test Set: {len(test_df)} slides")
    print(f"   • Normal: {(test_df['label'] == 0).sum()}")
    print(f"   • Tumor: {(test_df['label'] == 1).sum()}")
    
    return train_df, val_df, test_df


def save_csv_files(data_dir: Path, 
                   full_df: pd.DataFrame,
                   train_df: pd.DataFrame,
                   val_df: pd.DataFrame,
                   test_df: pd.DataFrame) -> None:
    """
    Save all CSV files.
    
    Args:
        data_dir: Path to data directory
        full_df: Full DataFrame with all slides
        train_df: Training split
        val_df: Validation split
        test_df: Test split
    """
    print("\n" + "="*80)
    print("SAVING CSV FILES")
    print("="*80 + "\n")
    
    # Save labels.csv
    labels_path = data_dir / 'labels.csv'
    full_df.to_csv(labels_path, index=False)
    print(f"✓ Saved {labels_path} ({len(full_df)} rows)")
    
    # Save labels_train.csv
    train_path = data_dir / 'labels_train.csv'
    train_df.to_csv(train_path, index=False)
    print(f"✓ Saved {train_path} ({len(train_df)} rows)")
    
    # Save labels_val.csv
    val_path = data_dir / 'labels_val.csv'
    val_df.to_csv(val_path, index=False)
    print(f"✓ Saved {val_path} ({len(val_df)} rows)")
    
    # Save labels_test.csv
    test_path = data_dir / 'labels_test.csv'
    test_df.to_csv(test_path, index=False)
    print(f"✓ Saved {test_path} ({len(test_df)} rows)")


def verify_labels(data_dir: Path) -> None:
    """
    Verify the generated label files.
    
    Args:
        data_dir: Path to data directory
    """
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80 + "\n")
    
    # Read all CSV files
    labels_df = pd.read_csv(data_dir / 'labels.csv')
    train_df = pd.read_csv(data_dir / 'labels_train.csv')
    val_df = pd.read_csv(data_dir / 'labels_val.csv')
    test_df = pd.read_csv(data_dir / 'labels_test.csv')
    
    # Verify class balance
    print("✓ Class Distribution in labels.csv:")
    print(labels_df['label'].value_counts().to_string())
    
    # Verify no overlap between splits
    train_slides = set(train_df['slide_name'])
    val_slides = set(val_df['slide_name'])
    test_slides = set(test_df['slide_name'])
    
    assert len(train_slides & val_slides) == 0, "Train-Val overlap detected!"
    assert len(train_slides & test_slides) == 0, "Train-Test overlap detected!"
    assert len(val_slides & test_slides) == 0, "Val-Test overlap detected!"
    print("\n✓ No overlap between train/val/test splits")
    
    # Verify all slides are accounted for
    total_split_slides = len(train_df) + len(val_df) + len(test_df)
    assert total_split_slides == len(labels_df), "Split count mismatch!"
    print(f"✓ All {len(labels_df)} slides accounted for in splits")
    
    print("\n" + "="*80)
    print("✅ LABELS REGENERATION COMPLETE")
    print("="*80)
    print("\n[READY] Dataset splits are ready for MIL training\n")


def main():
    """Main entry point."""
    data_dir = Path('data')
    
    print("\n" + "="*80)
    print("LABEL REGENERATION FROM FEATURES_TOPK")
    print("="*80)
    print(f"Data Directory: {data_dir.absolute()}\n")
    
    # Step 1: Scan features_topk directory
    full_df = scan_features_topk(data_dir)
    
    if len(full_df) == 0:
        print("\n❌ Error: No slides found in features_topk/")
        sys.exit(1)
    
    # Step 2: Create stratified splits
    train_df, val_df, test_df = create_stratified_splits(full_df)
    
    # Step 3: Save CSV files
    save_csv_files(data_dir, full_df, train_df, val_df, test_df)
    
    # Step 4: Verify
    verify_labels(data_dir)


if __name__ == "__main__":
    main()
