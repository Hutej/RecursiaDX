"""
YOLOv8 Inference Script for Blood Cell Detection
Detects RBC, WBC, and Platelets in blood smear images.

🔬 DISCLAIMER: This produces ESTIMATED counts per field of view.
   These are NOT clinical CBC values and should NOT replace laboratory analysis.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


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


def load_model(weights_path=None):
    """Load YOLOv8 model from weights."""
    if weights_path is None:
        # Default to best weights from training
        default_paths = [
            Path("runs/detect/bccd_blood_cells/weights/best.pt"),
            Path("runs/detect/train/weights/best.pt"),
            Path("best.pt"),
        ]
        for path in default_paths:
            if path.exists():
                weights_path = path
                break
    
    if weights_path is None or not Path(weights_path).exists():
        print("ERROR: Model weights not found!")
        print("Please train the model first using train.py, or specify weights path.")
        return None
    
    print(f"Loading model from: {weights_path}")
    model = YOLO(str(weights_path))
    return model


def run_inference(model, image_path, conf_threshold=0.25, save_output=True, output_dir=None):
    """
    Run inference on a blood smear image.
    
    Args:
        model: Loaded YOLOv8 model
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections
        save_output: Whether to save annotated image
        output_dir: Directory to save output (default: same as input)
    
    Returns:
        dict: Detection counts per class
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*60}")
    
    # Run inference
    results = model(str(image_path), conf=conf_threshold, verbose=False)[0]
    
    # Count detections per class
    counts = {"RBC": 0, "WBC": 0, "Platelets": 0}
    
    # Load image for visualization
    image = cv2.imread(str(image_path))
    
    # Process detections
    boxes = results.boxes
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = CLASS_NAMES.get(class_id, "Unknown")
        
        # Update count
        if class_name in counts:
            counts[class_name] += 1
        
        # Draw bounding box
        color = CLASS_COLORS.get(class_id, (128, 128, 128))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add summary text to image
    summary_y = 30
    cv2.putText(image, "ESTIMATED COUNTS (per field of view)", (10, summary_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    summary_y += 30
    for class_name, count in counts.items():
        color = CLASS_COLORS.get(list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(class_name)], (0, 0, 0))
        cv2.putText(image, f"{class_name}: {count}", (10, summary_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        summary_y += 25
    
    # Add disclaimer
    disclaimer = "Screening-level only - NOT clinical CBC"
    cv2.putText(image, disclaimer, (10, image.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Save output image
    if save_output:
        if output_dir is None:
            output_dir = Path("inference_results")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{image_path.stem}_detected_{timestamp}.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"\nOutput saved: {output_path}")
    
    # Print results
    print("\n" + "=" * 50)
    print("=== ESTIMATED COUNTS (per field of view) ===")
    print("=" * 50)
    print(f"  RBC detected:       {counts['RBC']}")
    print(f"  WBC detected:       {counts['WBC']}")
    print(f"  Platelets detected: {counts['Platelets']}")
    print("=" * 50)
    print("\n⚠️  IMPORTANT DISCLAIMER:")
    print("    These are ESTIMATED counts from visual detection.")
    print("    They represent counts per field of view ONLY.")
    print("    This is NOT a clinical Complete Blood Count (CBC).")
    print("    Do NOT use for medical diagnosis or treatment.")
    print("    For clinical analysis, use certified laboratory equipment.")
    print()
    
    return counts


def run_batch_inference(model, image_dir, conf_threshold=0.25, output_dir=None):
    """Run inference on all images in a directory."""
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        print(f"ERROR: Directory not found: {image_dir}")
        return
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"No images found in {image_dir}")
        return
    
    print(f"\nFound {len(images)} images in {image_dir}")
    
    # Process all images
    all_counts = {"RBC": 0, "WBC": 0, "Platelets": 0}
    for img_path in images:
        counts = run_inference(model, img_path, conf_threshold, save_output=True, output_dir=output_dir)
        if counts:
            for key in all_counts:
                all_counts[key] += counts[key]
    
    # Print aggregate statistics
    print("\n" + "=" * 60)
    print("AGGREGATE STATISTICS (across all processed images)")
    print("=" * 60)
    print(f"  Total RBC detected:       {all_counts['RBC']}")
    print(f"  Total WBC detected:       {all_counts['WBC']}")
    print(f"  Total Platelets detected: {all_counts['Platelets']}")
    print(f"  Images processed:         {len(images)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Blood Cell Detection Inference")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--dir", type=str, help="Path to directory of images")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="inference_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.weights)
    if model is None:
        return
    
    # Run inference
    if args.image:
        run_inference(model, args.image, args.conf, save_output=True, output_dir=args.output)
    elif args.dir:
        run_batch_inference(model, args.dir, args.conf, output_dir=args.output)
    else:
        print("Please specify --image or --dir for inference.")
        print("\nExamples:")
        print('  python infer.py --image "BCCD_Dataset-master/BCCD/JPEGImages/BloodImage_00007.jpg"')
        print('  python infer.py --dir "test_images/"')


if __name__ == "__main__":
    main()
