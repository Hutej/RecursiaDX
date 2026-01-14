# 🔬 Gigapixel Histopathology Analysis Pipeline
# End-to-end processing: Tiling → Classification → Aggregation → Visualization

import os
import time
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

from .tiling import GigapixelTiler, PatchExtractor
from .classifier import PatchClassifier
from .aggregation import HeatmapGenerator, LesionDetector, calculate_tumor_burden
from .attention import MultiScaleAttention, aggregate_patch_attentions


class HistopathologyPipeline:
    """
    Complete pipeline for gigapixel histopathology image analysis.
    
    Workflow:
    1. Tile gigapixel image into patches
    2. Classify each patch (tumor/normal)
    3. Aggregate predictions into heatmap
    4. Detect and segment lesions
    5. Generate interpretable visualizations
    """
    
    def __init__(
        self,
        model_path: str,
        patch_size: int = 224,
        overlap: float = 0.25,
        detection_threshold: float = 0.5,
        device: str = 'cuda',
        verbose: bool = True
    ):
        """
        Initialize pipeline.
        
        Args:
            model_path: Path to trained model (.pth file)
            patch_size: Size of patches to extract
            overlap: Overlap between patches (0-1)
            detection_threshold: Probability threshold for tumor detection
            device: 'cuda' or 'cpu'
            verbose: Print progress
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.detection_threshold = detection_threshold
        self.verbose = verbose
        
        # Initialize components
        if verbose:
            print("🔬 Initializing Histopathology Analysis Pipeline")
            print("=" * 70)
        
        # Tiler
        self.tiler = GigapixelTiler(
            patch_size=patch_size,
            overlap=overlap,
            scales=[1.0],  # Single scale for efficiency
            tissue_threshold=0.85
        )
        
        # Classifier
        self.classifier = PatchClassifier(
            model_path=model_path,
            device=device,
            threshold=detection_threshold
        )
        
        if verbose:
            print("✅ Pipeline initialized successfully\n")
    
    def process_image(
        self,
        image_path: str,
        output_dir: str = None,
        save_heatmap: bool = True,
        save_overlay: bool = True,
        save_detections: bool = True,
        batch_size: int = 32
    ) -> Dict:
        """
        Process a gigapixel histopathology image.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            save_heatmap: Save probability heatmap
            save_overlay: Save heatmap overlaid on original
            save_detections: Save detected lesion bounding boxes
            batch_size: Batch size for inference
            
        Returns:
            Dictionary with results and statistics
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"🔬 Processing: {os.path.basename(image_path)}")
            print("=" * 70)
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.splitext(image_path)[0] + "_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image to get size
        img = Image.open(image_path)
        image_size = img.size  # (width, height)
        
        if self.verbose:
            print(f"📏 Image size: {image_size[0]}x{image_size[1]} pixels")
        
        # Step 1: Extract and classify patches
        if self.verbose:
            print("\n📦 Step 1: Extracting and classifying patches...")
        
        patch_predictions = []
        patch_positions = []
        patch_batch = []
        position_batch = []
        
        # Initialize heatmap generator
        heatmap_gen = HeatmapGenerator(
            image_size=image_size,
            patch_size=self.patch_size,
            aggregation_method='weighted_average',
            smoothing_sigma=3.0
        )
        
        # Process patches
        patch_count = 0
        for patch, patch_info in self.tiler.extract_patches(image_path, save_patches=False):
            patch_batch.append(patch)
            position_batch.append((patch_info.x, patch_info.y))
            
            # Process batch
            if len(patch_batch) >= batch_size:
                predictions = self.classifier.predict_batch(patch_batch, batch_size=batch_size)
                
                # Add to heatmap
                for pred, (x, y) in zip(predictions, position_batch):
                    heatmap_gen.add_patch_prediction(
                        x, y,
                        pred['tumor_probability'],
                        pred['confidence']
                    )
                    patch_predictions.append(pred)
                    patch_positions.append((x, y))
                
                patch_count += len(patch_batch)
                patch_batch = []
                position_batch = []
        
        # Process remaining patches
        if patch_batch:
            predictions = self.classifier.predict_batch(patch_batch, batch_size=batch_size)
            
            for pred, (x, y) in zip(predictions, position_batch):
                heatmap_gen.add_patch_prediction(
                    x, y,
                    pred['tumor_probability'],
                    pred['confidence']
                )
                patch_predictions.append(pred)
                patch_positions.append((x, y))
            
            patch_count += len(patch_batch)
        
        if self.verbose:
            print(f"✅ Classified {patch_count} patches")
        
        # Step 2: Generate heatmap
        if self.verbose:
            print("\n🗺️  Step 2: Generating probability heatmap...")
        
        heatmap = heatmap_gen.generate_heatmap(apply_smoothing=True)
        
        # Calculate statistics
        tumor_patches = sum(1 for p in patch_predictions if p['class_id'] == 1)
        tumor_ratio = tumor_patches / len(patch_predictions) if patch_predictions else 0
        avg_tumor_prob = np.mean([p['tumor_probability'] for p in patch_predictions])
        
        if self.verbose:
            print(f"✅ Heatmap generated")
            print(f"   Tumor patches: {tumor_patches}/{len(patch_predictions)} ({tumor_ratio*100:.1f}%)")
            print(f"   Avg tumor probability: {avg_tumor_prob*100:.1f}%")
        
        # Step 3: Detect lesions
        if self.verbose:
            print("\n🎯 Step 3: Detecting lesions...")
        
        detector = LesionDetector(
            detection_threshold=self.detection_threshold,
            min_lesion_size=100
        )
        lesions = detector.detect_lesions(heatmap, return_masks=True)
        
        if self.verbose:
            print(f"✅ Detected {len(lesions)} lesions")
            if lesions:
                print(f"   Top lesion confidence: {lesions[0]['avg_confidence']*100:.1f}%")
                print(f"   Largest lesion area: {lesions[0]['area']} pixels")
        
        # Calculate tumor burden
        tumor_metrics = calculate_tumor_burden(lesions, image_size)
        
        # Step 4: Generate visualizations
        if self.verbose:
            print("\n🎨 Step 4: Generating visualizations...")
        
        # Load original image as numpy array
        img_array = np.array(img.resize((1024, 1024)))  # Resize for visualization
        heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((1024, 1024))) / 255.0
        
        # Create visualizations
        colored_heatmap = heatmap_gen.apply_colormap(heatmap_resized)
        overlay = heatmap_gen.create_overlay(img_array, heatmap_resized, alpha=0.4)
        
        # Draw detections
        # Resize lesion bounding boxes for visualization
        scale_x = 1024 / image_size[0]
        scale_y = 1024 / image_size[1]
        scaled_lesions = []
        for lesion in lesions:
            x_min, y_min, x_max, y_max = lesion['bbox']
            scaled_lesion = lesion.copy()
            scaled_lesion['bbox'] = (
                int(x_min * scale_x),
                int(y_min * scale_y),
                int(x_max * scale_x),
                int(y_max * scale_y)
            )
            scaled_lesions.append(scaled_lesion)
        
        detections_img = detector.draw_detections(img_array, scaled_lesions)
        
        # Save individual results
        if save_heatmap:
            Image.fromarray(colored_heatmap).save(os.path.join(output_dir, "heatmap.png"))
        if save_overlay:
            Image.fromarray(overlay).save(os.path.join(output_dir, "overlay.png"))
        if save_detections:
            Image.fromarray(detections_img).save(os.path.join(output_dir, "detections.png"))
        
        # Create comprehensive report
        self._generate_report(
            img_array,
            colored_heatmap,
            overlay,
            detections_img,
            heatmap_resized,
            lesions,
            tumor_metrics,
            patch_predictions,
            output_dir
        )
        
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n✅ Analysis complete in {elapsed_time:.1f}s")
            print(f"📁 Results saved to: {output_dir}")
        
        # Return results
        return {
            'image_path': image_path,
            'image_size': image_size,
            'num_patches': len(patch_predictions),
            'tumor_patches': tumor_patches,
            'tumor_ratio': tumor_ratio,
            'avg_tumor_probability': avg_tumor_prob,
            'num_lesions': len(lesions),
            'lesions': lesions,
            'tumor_burden': tumor_metrics,
            'heatmap': heatmap,
            'processing_time': elapsed_time,
            'output_dir': output_dir
        }
    
    def _generate_report(
        self,
        original_img,
        heatmap,
        overlay,
        detections,
        heatmap_array,
        lesions,
        tumor_metrics,
        predictions,
        output_dir
    ):
        """Generate comprehensive analysis report"""
        fig = plt.figure(figsize=(20, 12))
        
        # Original image
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(original_img)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Probability heatmap
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(heatmap)
        ax2.set_title('Tumor Probability Heatmap', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Overlay
        ax3 = plt.subplot(2, 3, 3)
        ax3.imshow(overlay)
        ax3.set_title('Heatmap Overlay', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # Detected lesions
        ax4 = plt.subplot(2, 3, 4)
        ax4.imshow(detections)
        ax4.set_title(f'Detected Lesions ({len(lesions)})', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        # Histogram
        ax5 = plt.subplot(2, 3, 5)
        tumor_probs = [p['tumor_probability'] for p in predictions]
        ax5.hist(tumor_probs, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax5.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        ax5.set_xlabel('Tumor Probability', fontsize=12)
        ax5.set_ylabel('Number of Patches', fontsize=12)
        ax5.set_title('Probability Distribution', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
╔══════════════════════════════════╗
║      ANALYSIS SUMMARY            ║
╚══════════════════════════════════╝

📊 PATCH STATISTICS:
  • Total patches: {len(predictions)}
  • Tumor patches: {sum(1 for p in predictions if p['class_id']==1)}
  • Normal patches: {sum(1 for p in predictions if p['class_id']==0)}
  • Tumor ratio: {tumor_metrics.get('tumor_burden_percentage', 0):.2f}%

🎯 LESION DETECTION:
  • Lesions detected: {len(lesions)}
  • Tumor burden: {tumor_metrics.get('tumor_burden_percentage', 0):.2f}%
  • Largest lesion: {tumor_metrics.get('largest_lesion_area', 0)} px²

📈 CONFIDENCE METRICS:
  • Avg probability: {np.mean(tumor_probs)*100:.1f}%
  • Max probability: {np.max(tumor_probs)*100:.1f}%
  • Min probability: {np.min(tumor_probs)*100:.1f}%

🏥 DIAGNOSIS:
  {'⚠️  TUMOR DETECTED' if len(lesions) > 0 else '✅ NO TUMOR DETECTED'}
        """
        
        ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.suptitle('🔬 Gigapixel Histopathology Analysis Report',
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Save
        report_path = os.path.join(output_dir, 'analysis_report.png')
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"✅ Report saved: {report_path}")


if __name__ == "__main__":
    print("🔬 Gigapixel Histopathology Pipeline")
    print("=" * 70)
    print("\nThis is the main pipeline module.")
    print("Use example_pipeline.py to run analysis on your images.")
