"""
YOLOv8 Training Script for Blood Cell Detection
Trains YOLOv8s on BCCD dataset for RBC, WBC, and Platelet detection.

🔬 This model produces ESTIMATED COUNTS per field of view, NOT clinical CBC values.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(epochs=200, batch_size=-1, img_size=640, resume=False):
    """
    Train YOLOv8s model on BCCD blood cell dataset.
    
    Args:
        epochs: Number of training epochs (default: 200)
        batch_size: Batch size (-1 for auto, maximizes GPU utilization)
        img_size: Input image size (default: 640)
        resume: Resume from last checkpoint if True
    """
    print("=" * 70)
    print("YOLOv8 Blood Cell Detection Training")
    print("=" * 70)
    print("\n⚠️  DISCLAIMER: This model provides ESTIMATED counts per field of view.")
    print("    It is NOT a replacement for clinical CBC laboratory analysis.\n")
    
    # Project paths
    project_root = Path(__file__).parent
    dataset_yaml = project_root / "dataset.yaml"
    
    if not dataset_yaml.exists():
        print(f"ERROR: dataset.yaml not found at {dataset_yaml}")
        print("Please run prepare_dataset.py first to set up the dataset.")
        return None
    
    # Initialize model with YOLOv8s pretrained on COCO
    print("Loading YOLOv8s with COCO pretrained weights...")
    model = YOLO("yolov8s.pt")
    
    # Training configuration optimized for blood cell detection
    # - Mosaic augmentation helps with dense object scenes
    # - HSV augmentation for stain variation robustness
    # - Scale augmentation for size variation
    # - Flip augmentations (horizontal + vertical) for orientation invariance
    # - No extreme rotations (blood cells are roughly circular)
    
    print(f"\nTraining Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {'Auto (GPU memory optimized)' if batch_size == -1 else batch_size}")
    print(f"  - Image size: {img_size}")
    print(f"  - Mixed precision: Enabled (FP16)")
    print(f"  - Augmentations: Mosaic, HSV, Scale, Flips")
    print(f"  - Small object focus: Platelets awareness enabled")
    print()
    
    # Train the model
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch_size,  # -1 = auto batch size (maximizes GPU utilization)
        imgsz=img_size,
        
        # GPU Optimization
        device=0,  # Use first GPU
        amp=True,  # Mixed precision training (FP16)
        workers=4,  # DataLoader workers
        
        # Augmentations (optimized for blood smear images)
        mosaic=1.0,  # Mosaic augmentation
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        scale=0.5,    # Scale augmentation (+/- 50%)
        fliplr=0.5,   # Horizontal flip probability
        flipud=0.5,   # Vertical flip probability (blood cells are orientation-invariant)
        
        # Avoid extreme augmentations that could hurt small object detection
        degrees=0.0,     # No rotation (cells are roughly circular anyway)
        translate=0.1,   # Slight translation
        shear=0.0,       # No shear
        perspective=0.0, # No perspective
        
        # Training settings
        patience=50,      # Early stopping patience
        save=True,        # Save checkpoints
        save_period=-1,   # Save only best and last
        cache=True,       # Cache images for faster training
        exist_ok=True,    # Allow reusing project folder
        
        # Logging & metrics
        verbose=True,
        plots=True,       # Generate training plots
        
        # Project output
        project="runs/detect",
        name="bccd_blood_cells",
        
        # Resume from checkpoint if specified
        resume=resume,
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nBest weights saved to: runs/detect/bccd_blood_cells/weights/best.pt")
    print(f"Last weights saved to: runs/detect/bccd_blood_cells/weights/last.pt")
    print("\n📊 Key metrics to review:")
    print("   - mAP@0.5 (overall detection accuracy)")
    print("   - mAP@0.5:0.95 (strict accuracy across IoU thresholds)")
    print("   - Per-class recall (especially Platelets and WBC)")
    print("\n⚠️  REMINDER: Results are ESTIMATED counts per field of view only.")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Blood Cell Detection")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for auto)")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
