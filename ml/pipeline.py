# 🔬 Gigapixel Histopathology Analysis Pipeline
# End-to-end processing: Tiling → Classification → Aggregation → Visualization

import os
import time
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tqdm import tqdm
import json
import base64
from io import BytesIO

try:
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from tiling import GigapixelTiler, PatchExtractor
from classifier import PatchClassifier
from aggregation import HeatmapGenerator, LesionDetector, calculate_tumor_burden
from attention import MultiScaleAttention, aggregate_patch_attentions


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
    
    def generate_matplotlib_heatmap(
        self,
        image_path: str,
        output_path: str = None,
        heatmap_type: str = 'tumor_probability',
        colormap: str = 'hot',
        figsize: Tuple[int, int] = (12, 8),
        show_colorbar: bool = True,
        show_overlay: bool = True,
        overlay_alpha: float = 0.6,
        dpi: int = 300,
        return_fig: bool = False
    ) -> Dict:
        """
        Generate matplotlib-style heatmap visualization.
        
        Args:
            image_path: Path to input image
            output_path: Path to save heatmap image
            heatmap_type: Type of heatmap ('tumor_probability', 'confidence', 'risk')
            colormap: Matplotlib colormap name
            figsize: Figure size in inches
            show_colorbar: Whether to show colorbar
            show_overlay: Whether to overlay on original image
            overlay_alpha: Transparency of heatmap overlay
            dpi: Image resolution
            return_fig: Whether to return matplotlib figure
            
        Returns:
            Dictionary with heatmap information and optionally matplotlib figure
        """
        try:
            # Process image
            results = self.process_image(
                image_path,
                output_dir=None,
                save_heatmap=False,
                save_overlay=False,
                save_detections=False
            )
            
            # Generate heatmap data
            heatmap_array = self._create_heatmap_array(results, heatmap_type)
            
            # Create matplotlib figure
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
            
            if show_overlay:
                # Load and display original image
                original_img = Image.open(image_path)
                original_img = original_img.resize(heatmap_array.shape[::-1])
                ax.imshow(np.array(original_img), alpha=1.0)
                
                # Overlay heatmap
                im = ax.imshow(
                    heatmap_array, 
                    cmap=colormap, 
                    alpha=overlay_alpha,
                    interpolation='bilinear'
                )
            else:
                # Show only heatmap
                im = ax.imshow(
                    heatmap_array, 
                    cmap=colormap,
                    interpolation='bilinear'
                )
            
            # Customize plot
            ax.set_title(f'{heatmap_type.replace("_", " ").title()} Heatmap', 
                        fontsize=16, fontweight='bold')
            ax.axis('off')
            
            # Add colorbar
            if show_colorbar:
                cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
                cbar.set_label(self._get_colorbar_label(heatmap_type), 
                              fontsize=12, fontweight='bold')
                cbar.ax.tick_params(labelsize=10)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if output path provided
            if output_path:
                plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                if self.verbose:
                    print(f"✅ Matplotlib heatmap saved: {output_path}")
            
            # Prepare return data
            result_data = {
                'success': True,
                'heatmap_info': {
                    'type': heatmap_type,
                    'colormap': colormap,
                    'shape': heatmap_array.shape,
                    'min_value': float(np.min(heatmap_array)),
                    'max_value': float(np.max(heatmap_array)),
                    'mean_value': float(np.mean(heatmap_array))
                },
                'image_info': {
                    'path': image_path,
                    'patches_analyzed': len(results.get('predictions', [])),
                    'processing_time': results.get('processing_time', 0)
                },
                'analytics': self._extract_analytics_data(results)
            }
            
            if output_path:
                result_data['output_path'] = output_path
            
            if return_fig:
                result_data['figure'] = fig
            else:
                plt.close(fig)
            
            return result_data
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_multiple_heatmaps(
        self,
        image_path: str,
        output_dir: str,
        heatmap_types: List[str] = None,
        colormaps: List[str] = None,
        create_comparison: bool = True
    ) -> Dict:
        """
        Generate multiple heatmap visualizations for comparison.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save heatmaps
            heatmap_types: List of heatmap types to generate
            colormaps: List of colormaps to use
            create_comparison: Whether to create comparison figure
            
        Returns:
            Dictionary with paths to generated heatmaps
        """
        if heatmap_types is None:
            heatmap_types = ['tumor_probability', 'confidence', 'risk_score']
        
        if colormaps is None:
            colormaps = ['hot', 'viridis', 'plasma', 'inferno']
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Process image once
            results = self.process_image(
                image_path,
                output_dir=None,
                save_heatmap=False,
                save_overlay=False,
                save_detections=False
            )
            
            generated_heatmaps = {}
            
            # Generate individual heatmaps
            for htype in heatmap_types:
                for cmap in colormaps:
                    filename = f"heatmap_{htype}_{cmap}.png"
                    output_path = os.path.join(output_dir, filename)
                    
                    result = self.generate_matplotlib_heatmap(
                        image_path,
                        output_path=output_path,
                        heatmap_type=htype,
                        colormap=cmap,
                        show_overlay=True
                    )
                    
                    if result['success']:
                        generated_heatmaps[f"{htype}_{cmap}"] = output_path
            
            # Create comparison figure
            if create_comparison and len(heatmap_types) > 1:
                comparison_path = os.path.join(output_dir, "heatmap_comparison.png")
                self._create_comparison_figure(
                    results, heatmap_types, comparison_path
                )
                generated_heatmaps['comparison'] = comparison_path
            
            return {
                'success': True,
                'generated_heatmaps': generated_heatmaps,
                'output_directory': output_dir,
                'total_generated': len(generated_heatmaps)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_heatmap_array(
        self, 
        results: Dict, 
        heatmap_type: str
    ) -> np.ndarray:
        """Create heatmap array from processing results."""
        predictions = results.get('predictions', [])
        if not predictions:
            raise ValueError("No predictions found in results")
        
        # Extract coordinates and values
        coords = []
        values = []
        
        for pred in predictions:
            coords.append([pred['x'], pred['y']])
            
            if heatmap_type == 'tumor_probability':
                values.append(pred.get('tumor_prob', pred.get('confidence', 0)))
            elif heatmap_type == 'confidence':
                values.append(pred.get('confidence', 0))
            elif heatmap_type == 'risk_score':
                values.append(self._calculate_patch_risk_score(pred))
            else:
                values.append(pred.get('confidence', 0))
        
        coords = np.array(coords)
        values = np.array(values)
        
        # Create grid
        img_size = results.get('image_size', (2048, 2048))
        grid_size = (min(512, img_size[1]//4), min(512, img_size[0]//4))
        
        # Interpolate values to grid
        if not SCIPY_AVAILABLE:
            # Simple grid assignment without interpolation
            heatmap = np.zeros(grid_size)
            for coord, value in zip(coords, values):
                x_idx = int((coord[0] / img_size[0]) * grid_size[0])
                y_idx = int((coord[1] / img_size[1]) * grid_size[1])
                x_idx = max(0, min(x_idx, grid_size[0] - 1))
                y_idx = max(0, min(y_idx, grid_size[1] - 1))
                heatmap[y_idx, x_idx] = value
        else:
            # Use scipy for proper interpolation
            # Create coordinate grids
            xi = np.linspace(0, img_size[0], grid_size[0])
            yi = np.linspace(0, img_size[1], grid_size[1])
            xi, yi = np.meshgrid(xi, yi)
            
            # Interpolate
            heatmap = griddata(
                coords, values, (xi, yi), 
                method='cubic', fill_value=0
            )
        
        # Ensure valid range
        heatmap = np.clip(heatmap, 0, 1)
        
        return heatmap
    
    def _create_comparison_figure(
        self,
        results: Dict,
        heatmap_types: List[str],
        output_path: str
    ):
        """Create comparison figure with multiple heatmaps."""
        n_types = len(heatmap_types)
        fig, axes = plt.subplots(2, n_types, figsize=(4*n_types, 8), dpi=150)
        
        if n_types == 1:
            axes = axes.reshape(2, 1)
        
        colormaps = ['hot', 'viridis', 'plasma', 'inferno'][:n_types]
        
        for i, (htype, cmap) in enumerate(zip(heatmap_types, colormaps)):
            # Generate heatmap array
            heatmap_array = self._create_heatmap_array(results, htype)
            
            # Top row: heatmap only
            im1 = axes[0, i].imshow(heatmap_array, cmap=cmap, interpolation='bilinear')
            axes[0, i].set_title(f'{htype.replace("_", " ").title()}', fontweight='bold')
            axes[0, i].axis('off')
            
            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=axes[0, i], shrink=0.8)
            cbar1.set_label(self._get_colorbar_label(htype), fontsize=10)
            
            # Bottom row: with overlay (if original image available)
            im2 = axes[1, i].imshow(heatmap_array, cmap=cmap, alpha=0.7, interpolation='bilinear')
            axes[1, i].set_title(f'{htype.replace("_", " ").title()} Overlay', fontweight='bold')
            axes[1, i].axis('off')
            
            # Add colorbar
            cbar2 = plt.colorbar(im2, ax=axes[1, i], shrink=0.8)
            cbar2.set_label(self._get_colorbar_label(htype), fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        if self.verbose:
            print(f"✅ Comparison figure saved: {output_path}")
    
    def _get_colorbar_label(self, heatmap_type: str) -> str:
        """Get appropriate colorbar label for heatmap type."""
        labels = {
            'tumor_probability': 'Tumor Probability',
            'confidence': 'Prediction Confidence',
            'risk_score': 'Risk Score',
            'attention': 'Attention Weight'
        }
        return labels.get(heatmap_type, 'Value')
    
    def _calculate_patch_risk_score(self, prediction: Dict) -> float:
        """Calculate risk score for a patch prediction."""
        tumor_prob = prediction.get('tumor_prob', prediction.get('confidence', 0))
        confidence = prediction.get('confidence', 0.5)
        
        # Weighted risk score
        risk_score = (tumor_prob * 0.8) + (confidence * 0.2)
        return np.clip(risk_score, 0, 1)
    
    def create_interactive_heatmap(
        self,
        image_path: str,
        output_path: str = None,
        heatmap_type: str = 'tumor_probability'
    ) -> Dict:
        """
        Create interactive heatmap using plotly.
        
        Args:
            image_path: Path to input image
            output_path: Path to save interactive HTML
            heatmap_type: Type of heatmap to generate
            
        Returns:
            Dictionary with interactive heatmap data
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            # Process image
            results = self.process_image(
                image_path,
                output_dir=None,
                save_heatmap=False,
                save_overlay=False,
                save_detections=False
            )
            
            # Generate heatmap data
            heatmap_array = self._create_heatmap_array(results, heatmap_type)
            
            # Create interactive heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_array,
                colorscale='Hot',
                colorbar=dict(
                    title=self._get_colorbar_label(heatmap_type),
                    titleside='right'
                ),
                hoverongaps=False,
                hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'Interactive {heatmap_type.replace("_", " ").title()} Heatmap',
                xaxis_title='X Position',
                yaxis_title='Y Position',
                width=800,
                height=600
            )
            
            # Save interactive HTML
            if output_path:
                fig.write_html(output_path)
                if self.verbose:
                    print(f"✅ Interactive heatmap saved: {output_path}")
            
            return {
                'success': True,
                'interactive_data': fig.to_dict(),
                'output_path': output_path if output_path else None,
                'heatmap_info': {
                    'type': heatmap_type,
                    'shape': heatmap_array.shape,
                    'min_value': float(np.min(heatmap_array)),
                    'max_value': float(np.max(heatmap_array))
                }
            }
            
        except ImportError:
            return {
                'success': False,
                'error': 'plotly not installed. Install with: pip install plotly'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_dashboard_heatmap(
        self,
        image_path: str,
        output_format: str = 'base64',
        heatmap_type: str = 'confidence',
        resolution: Tuple[int, int] = (512, 512)
    ) -> Dict:
        """
        Generate heatmap specifically for AI analytics dashboard.
        
        Args:
            image_path: Path to input image
            output_format: 'base64', 'file', or 'array'
            heatmap_type: 'confidence', 'risk', 'attention', or 'tumor_probability'
            resolution: Output resolution for dashboard display
            
        Returns:
            Dictionary with heatmap data for dashboard
        """
        try:
            # Process image with minimal memory footprint
            results = self.process_image(
                image_path,
                output_dir=None,
                save_heatmap=False,
                save_overlay=False,
                save_detections=False,
                batch_size=16  # Smaller batch for dashboard
            )
            
            # Generate specific heatmap type
            heatmap_data = self._create_dashboard_heatmap(
                results, heatmap_type, resolution
            )
            
            # Format for dashboard consumption
            dashboard_data = {
                'success': True,
                'image_info': {
                    'path': image_path,
                    'original_size': results.get('image_size', (0, 0)),
                    'patches_analyzed': len(results.get('predictions', [])),
                    'processing_time': results.get('processing_time', 0)
                },
                'heatmap': {
                    'type': heatmap_type,
                    'resolution': resolution,
                    'data': heatmap_data,
                    'colormap': self._get_colormap_info(heatmap_type)
                },
                'analytics': self._extract_analytics_data(results),
                'timestamp': time.time()
            }
            
            if output_format == 'base64':
                dashboard_data['heatmap']['data'] = self._array_to_base64(heatmap_data)
            
            return dashboard_data
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _create_dashboard_heatmap(
        self, 
        results: Dict, 
        heatmap_type: str, 
        resolution: Tuple[int, int]
    ) -> np.ndarray:
        """Create specific heatmap type for dashboard."""
        predictions = results.get('predictions', [])
        if not predictions:
            return np.zeros(resolution)
        
        # Extract coordinates and values
        coords = [(p['x'], p['y']) for p in predictions]
        
        if heatmap_type == 'confidence':
            values = [p['confidence'] for p in predictions]
        elif heatmap_type == 'risk':
            values = [self._calculate_risk_score(p) for p in predictions]
        elif heatmap_type == 'tumor_probability':
            values = [p['tumor_probability'] for p in predictions]
        elif heatmap_type == 'attention':
            values = [p.get('attention_score', 0.5) for p in predictions]
        else:
            values = [p['confidence'] for p in predictions]
        
        # Create heatmap array
        heatmap = self._interpolate_heatmap(coords, values, resolution)
        return heatmap
    
    def _calculate_risk_score(self, prediction: Dict) -> float:
        """Calculate risk score for dashboard visualization."""
        tumor_prob = prediction.get('tumor_probability', 0)
        confidence = prediction.get('confidence', 0)
        
        # Risk scoring: high tumor probability + high confidence = high risk
        risk_score = (tumor_prob * 0.7) + (confidence * 0.3)
        return min(max(risk_score, 0.0), 1.0)
    
    def _interpolate_heatmap(
        self, 
        coords: List[Tuple[int, int]], 
        values: List[float], 
        resolution: Tuple[int, int]
    ) -> np.ndarray:
        """Interpolate patch values to create smooth heatmap."""
        from scipy.interpolate import griddata
        
        if not coords or not values:
            return np.zeros(resolution)
        
        # Create grid for interpolation
        x = np.linspace(0, max(c[0] for c in coords), resolution[1])
        y = np.linspace(0, max(c[1] for c in coords), resolution[0])
        grid_x, grid_y = np.meshgrid(x, y)
        
        # Interpolate values
        try:
            heatmap = griddata(
                coords, values, (grid_x, grid_y), 
                method='linear', fill_value=0
            )
            return np.nan_to_num(heatmap, 0)
        except:
            # Fallback: simple binning
            return self._simple_binning_heatmap(coords, values, resolution)
    
    def _simple_binning_heatmap(
        self, 
        coords: List[Tuple[int, int]], 
        values: List[float], 
        resolution: Tuple[int, int]
    ) -> np.ndarray:
        """Simple binning fallback for heatmap generation."""
        heatmap = np.zeros(resolution)
        
        if not coords:
            return heatmap
        
        max_x = max(c[0] for c in coords)
        max_y = max(c[1] for c in coords)
        
        for (x, y), value in zip(coords, values):
            # Map to grid coordinates
            grid_x = int((x / max_x) * (resolution[1] - 1)) if max_x > 0 else 0
            grid_y = int((y / max_y) * (resolution[0] - 1)) if max_y > 0 else 0
            
            # Ensure within bounds
            grid_x = max(0, min(grid_x, resolution[1] - 1))
            grid_y = max(0, min(grid_y, resolution[0] - 1))
            
            heatmap[grid_y, grid_x] = max(heatmap[grid_y, grid_x], value)
        
        return heatmap
    
    def _get_colormap_info(self, heatmap_type: str) -> Dict:
        """Get colormap information for dashboard."""
        colormaps = {
            'confidence': {
                'name': 'viridis',
                'description': 'Model confidence levels',
                'scale': 'linear',
                'range': [0, 1]
            },
            'risk': {
                'name': 'hot',
                'description': 'Risk assessment levels',
                'scale': 'linear',
                'range': [0, 1]
            },
            'tumor_probability': {
                'name': 'Reds',
                'description': 'Tumor probability',
                'scale': 'linear',
                'range': [0, 1]
            },
            'attention': {
                'name': 'plasma',
                'description': 'Model attention weights',
                'scale': 'linear',
                'range': [0, 1]
            }
        }
        return colormaps.get(heatmap_type, colormaps['confidence'])
    
    def _extract_analytics_data(self, results: Dict) -> Dict:
        """Extract key analytics for dashboard display."""
        predictions = results.get('predictions', [])
        if not predictions:
            return {}
        
        tumor_predictions = [p for p in predictions if p.get('class_id') == 1]
        tumor_probs = [p['tumor_probability'] for p in predictions]
        
        return {
            'total_patches': len(predictions),
            'tumor_patches': len(tumor_predictions),
            'normal_patches': len(predictions) - len(tumor_predictions),
            'tumor_percentage': (len(tumor_predictions) / len(predictions)) * 100,
            'average_confidence': np.mean([p['confidence'] for p in predictions]),
            'max_tumor_probability': np.max(tumor_probs) if tumor_probs else 0,
            'min_tumor_probability': np.min(tumor_probs) if tumor_probs else 0,
            'high_risk_patches': len([p for p in predictions if self._calculate_risk_score(p) > 0.7]),
            'lesions_detected': len(results.get('lesions', [])),
            'processing_time': results.get('processing_time', 0)
        }
    
    def _array_to_base64(self, array: np.ndarray) -> str:
        """Convert numpy array to base64 string for dashboard."""
        # Normalize to 0-255
        normalized = ((array - array.min()) / (array.max() - array.min() + 1e-8) * 255).astype(np.uint8)
        
        # Convert to PIL Image
        img = Image.fromarray(normalized, mode='L')
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def generate_real_time_analytics(self, image_path: str) -> Dict:
        """
        Generate real-time analytics for dashboard streaming.
        Optimized for quick updates and minimal processing.
        """
        try:
            # Quick analysis with reduced resolution
            start_time = time.time()
            
            # Process with minimal settings
            results = self.process_image(
                image_path,
                output_dir=None,
                save_heatmap=False,
                save_overlay=False,
                save_detections=False,
                batch_size=8  # Very small batch for speed
            )
            
            processing_time = time.time() - start_time
            
            # Extract key metrics
            predictions = results.get('predictions', [])
            analytics = self._extract_analytics_data(results)
            
            return {
                'success': True,
                'timestamp': time.time(),
                'processing_time': processing_time,
                'metrics': {
                    'total_patches': analytics.get('total_patches', 0),
                    'tumor_percentage': analytics.get('tumor_percentage', 0),
                    'average_confidence': analytics.get('average_confidence', 0),
                    'high_risk_patches': analytics.get('high_risk_patches', 0),
                    'status': 'tumor_detected' if analytics.get('tumor_percentage', 0) > 5 else 'normal'
                },
                'quick_heatmap': self._generate_quick_heatmap(predictions)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _generate_quick_heatmap(self, predictions: List[Dict], size: int = 64) -> str:
        """Generate a quick, low-resolution heatmap for real-time updates."""
        if not predictions:
            return self._array_to_base64(np.zeros((size, size)))
        
        coords = [(p['x'], p['y']) for p in predictions]
        values = [p['tumor_probability'] for p in predictions]
        
        heatmap = self._simple_binning_heatmap(coords, values, (size, size))
        return self._array_to_base64(heatmap)


if __name__ == "__main__":
    print("🔬 Gigapixel Histopathology Pipeline")
    print("=" * 70)
    print("\nThis is the main pipeline module.")
    print("Use example_pipeline.py to run analysis on your images.")
