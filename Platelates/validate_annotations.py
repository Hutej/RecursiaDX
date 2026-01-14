"""
Visual Validation Script for YOLO Annotations
Overlays bounding boxes on sample images to verify annotation conversion quality.
"""

import os
import random
from pathlib import Path

import cv2


# Class colors (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 0, 255),     # RBC - Red
    1: (255, 0, 0),     # WBC - Blue
    2: (0, 255, 255),   # Platelets - Yellow
}

CLASS_NAMES = {
    0: "RBC",
    1: "WBC", 
    2: "Platelets"
}


def validate_annotations(num_samples=10, dataset_dir="datasets", output_dir="validation_samples"):
    """
    Visualize YOLO annotations on sample images.
    
    Args:
        num_samples: Number of random samples to validate
        dataset_dir: Path to YOLO dataset directory
        output_dir: Path to save validation images
    """
    print("=" * 60)
    print("YOLO Annotation Visual Validation")
    print("=" * 60)
    
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all train/val images
    all_images = []
    for split in ['train', 'val']:
        img_dir = dataset_dir / 'images' / split
        if img_dir.exists():
            all_images.extend(list(img_dir.glob('*.jpg')))
    
    if not all_images:
        print("ERROR: No images found in dataset directory")
        print(f"  Expected: {dataset_dir}/images/train/ or {dataset_dir}/images/val/")
        return
    
    # Sample random images
    samples = random.sample(all_images, min(num_samples, len(all_images)))
    print(f"\nValidating {len(samples)} random samples...")
    
    for img_path in samples:
        # Determine split and find label file
        split = img_path.parent.name
        label_path = dataset_dir / 'labels' / split / f"{img_path.stem}.txt"
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Warning: Could not load {img_path}")
            continue
        
        h, w = image.shape[:2]
        
        # Load and draw annotations
        annotation_count = {"RBC": 0, "WBC": 0, "Platelets": 0}
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    box_w = float(parts[3]) * w
                    box_h = float(parts[4]) * h
                    
                    # Convert to corner coordinates
                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)
                    
                    # Draw box
                    color = CLASS_COLORS.get(class_id, (128, 128, 128))
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Update count
                    class_name = CLASS_NAMES.get(class_id, "Unknown")
                    if class_name in annotation_count:
                        annotation_count[class_name] += 1
        
        # Add legend
        legend_y = 25
        cv2.putText(image, f"File: {img_path.name}", (10, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        legend_y += 25
        for class_name, count in annotation_count.items():
            class_id = list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(class_name)]
            color = CLASS_COLORS.get(class_id, (0, 0, 0))
            cv2.putText(image, f"{class_name}: {count}", (10, legend_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            legend_y += 22
        
        # Save
        output_path = output_dir / f"validated_{img_path.name}"
        cv2.imwrite(str(output_path), image)
        
        total = sum(annotation_count.values())
        print(f"  ✓ {img_path.name}: {total} annotations "
              f"(RBC:{annotation_count['RBC']}, WBC:{annotation_count['WBC']}, "
              f"Platelets:{annotation_count['Platelets']})")
    
    print(f"\n✅ Validation images saved to: {output_dir}")
    print("   Review these images to verify annotation quality.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate YOLO annotations visually")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to validate")
    parser.add_argument("--dataset", type=str, default="datasets", help="Dataset directory")
    parser.add_argument("--output", type=str, default="validation_samples", help="Output directory")
    
    args = parser.parse_args()
    validate_annotations(args.samples, args.dataset, args.output)
