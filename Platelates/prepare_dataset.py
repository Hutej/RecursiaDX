"""
VOC to YOLO Dataset Converter for BCCD Blood Cell Dataset
Converts Pascal VOC XML annotations to YOLO format and organizes dataset structure.
"""

import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

# Class mapping (as specified in requirements)
CLASS_MAPPING = {
    'RBC': 0,
    'WBC': 1,
    'Platelets': 2
}

# Paths
PROJECT_ROOT = Path(__file__).parent
BCCD_ROOT = PROJECT_ROOT / "BCCD_Dataset-master" / "BCCD"
ANNOTATIONS_DIR = BCCD_ROOT / "Annotations"
IMAGES_DIR = BCCD_ROOT / "JPEGImages"
IMAGESETS_DIR = BCCD_ROOT / "ImageSets" / "Main"
OUTPUT_DIR = PROJECT_ROOT / "datasets"


def parse_voc_annotation(xml_path):
    """Parse a Pascal VOC XML annotation file and return YOLO format annotations."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    yolo_annotations = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        
        # Skip unknown classes
        if class_name not in CLASS_MAPPING:
            print(f"Warning: Unknown class '{class_name}' in {xml_path}")
            continue
        
        class_id = CLASS_MAPPING[class_name]
        
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # Convert to YOLO format (normalized center coordinates and dimensions)
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        
        # Clamp values to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        box_width = max(0, min(1, box_width))
        box_height = max(0, min(1, box_height))
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
    
    return yolo_annotations


def read_split_file(split_file):
    """Read image names from a split file (train.txt, val.txt, test.txt)."""
    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def convert_dataset():
    """Convert BCCD dataset from VOC to YOLO format."""
    print("=" * 60)
    print("BCCD Dataset Converter: VOC to YOLO Format")
    print("=" * 60)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {
        'train': {'images': 0, 'RBC': 0, 'WBC': 0, 'Platelets': 0},
        'val': {'images': 0, 'RBC': 0, 'WBC': 0, 'Platelets': 0},
        'test': {'images': 0, 'RBC': 0, 'WBC': 0, 'Platelets': 0}
    }
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_file = IMAGESETS_DIR / f"{split}.txt"
        if not split_file.exists():
            print(f"Warning: {split_file} not found, skipping {split} split")
            continue
        
        image_names = read_split_file(split_file)
        print(f"\nProcessing {split} split: {len(image_names)} images")
        
        for img_name in image_names:
            # Source paths
            xml_path = ANNOTATIONS_DIR / f"{img_name}.xml"
            img_path = IMAGES_DIR / f"{img_name}.jpg"
            
            # Check if files exist
            if not xml_path.exists():
                print(f"  Warning: Annotation not found for {img_name}")
                continue
            if not img_path.exists():
                print(f"  Warning: Image not found for {img_name}")
                continue
            
            # Parse annotations
            yolo_annotations = parse_voc_annotation(xml_path)
            
            # Copy image
            dst_img_path = OUTPUT_DIR / 'images' / split / f"{img_name}.jpg"
            shutil.copy2(img_path, dst_img_path)
            
            # Write YOLO label file
            dst_label_path = OUTPUT_DIR / 'labels' / split / f"{img_name}.txt"
            with open(dst_label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            # Update statistics
            stats[split]['images'] += 1
            for ann in yolo_annotations:
                class_id = int(ann.split()[0])
                if class_id == 0:
                    stats[split]['RBC'] += 1
                elif class_id == 1:
                    stats[split]['WBC'] += 1
                elif class_id == 2:
                    stats[split]['Platelets'] += 1
    
    # Print statistics
    print("\n" + "=" * 60)
    print("DATASET CONVERSION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"{'Split':<10} {'Images':<10} {'RBC':<10} {'WBC':<10} {'Platelets':<10}")
    print("-" * 50)
    
    total = {'images': 0, 'RBC': 0, 'WBC': 0, 'Platelets': 0}
    for split in ['train', 'val', 'test']:
        s = stats[split]
        print(f"{split:<10} {s['images']:<10} {s['RBC']:<10} {s['WBC']:<10} {s['Platelets']:<10}")
        for key in total:
            total[key] += s[key]
    
    print("-" * 50)
    print(f"{'TOTAL':<10} {total['images']:<10} {total['RBC']:<10} {total['WBC']:<10} {total['Platelets']:<10}")
    print("-" * 50)
    
    return stats


if __name__ == "__main__":
    convert_dataset()
