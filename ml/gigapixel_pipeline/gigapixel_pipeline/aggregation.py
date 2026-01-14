# 🗺️ Aggregation and Heatmap Generation Module
# Convert patch predictions into whole-slide interpretable heatmaps

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Dict, Optional


class HeatmapGenerator:
    """
    Generate interpretable heatmaps from patch-level predictions.
    
    Features:
    - Spatial aggregation of overlapping predictions
    - Gaussian smoothing for visual clarity
    - Multi-level thresholding
    - Colormap application
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: int = 224,
        aggregation_method: str = 'weighted_average',  # 'max', 'average', 'weighted_average'
        smoothing_sigma: float = 2.0,
        colormap: str = 'jet'
    ):
        self.image_size = image_size  # (width, height)
        self.patch_size = patch_size
        self.aggregation_method = aggregation_method
        self.smoothing_sigma = smoothing_sigma
        self.colormap = colormap
        
        # Initialize heatmap arrays
        self.height, self.width = image_size[1], image_size[0]
        self.probability_map = np.zeros((self.height, self.width), dtype=np.float32)
        self.confidence_map = np.zeros((self.height, self.width), dtype=np.float32)
        self.count_map = np.zeros((self.height, self.width), dtype=np.int32)
    
    def add_patch_prediction(
        self,
        x: int,
        y: int,
        probability: float,
        confidence: float,
        patch_size: Optional[int] = None
    ):
        """
        Add a patch prediction to the heatmap.
        
        Args:
            x, y: Top-left corner of patch
            probability: Tumor probability [0, 1]
            confidence: Prediction confidence [0, 100]
            patch_size: Override default patch size
        """
        size = patch_size or self.patch_size
        
        # Ensure within bounds
        x_end = min(x + size, self.width)
        y_end = min(y + size, self.height)
        
        # Add to maps
        if self.aggregation_method == 'max':
            self.probability_map[y:y_end, x:x_end] = np.maximum(
                self.probability_map[y:y_end, x:x_end],
                probability
            )
        else:  # average or weighted_average
            if self.aggregation_method == 'weighted_average':
                weight = confidence / 100.0  # Normalize confidence to [0, 1]
            else:
                weight = 1.0
            
            self.probability_map[y:y_end, x:x_end] += probability * weight
            self.confidence_map[y:y_end, x:x_end] += confidence * weight
            self.count_map[y:y_end, x:x_end] += 1
    
    def generate_heatmap(
        self,
        apply_smoothing: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate final heatmap from accumulated predictions.
        
        Returns:
            heatmap: (H, W) numpy array with values in [0, 1]
        """
        # Average overlapping predictions
        heatmap = np.divide(
            self.probability_map,
            self.count_map,
            out=np.zeros_like(self.probability_map),
            where=self.count_map > 0
        )
        
        # Apply Gaussian smoothing
        if apply_smoothing and self.smoothing_sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=self.smoothing_sigma)
        
        # Normalize
        if normalize:
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap
    
    def apply_colormap(
        self,
        heatmap: np.ndarray,
        colormap: Optional[str] = None
    ) -> np.ndarray:
        """
        Apply colormap to heatmap for visualization.
        
        Returns:
            colored_heatmap: (H, W, 3) RGB image
        """
        cmap_name = colormap or self.colormap
        cmap = cm.get_cmap(cmap_name)
        colored = cmap(heatmap)[:, :, :3]  # Remove alpha
        return (colored * 255).astype(np.uint8)
    
    def create_overlay(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            original_image: (H, W, 3) RGB image
            heatmap: (H, W) probability map
            alpha: Overlay transparency
            
        Returns:
            overlay: (H, W, 3) RGB image with heatmap overlay
        """
        # Resize heatmap to match image if needed
        if heatmap.shape != original_image.shape[:2]:
            heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
            heatmap_pil = heatmap_pil.resize(
                (original_image.shape[1], original_image.shape[0]),
                Image.BILINEAR
            )
            heatmap = np.array(heatmap_pil) / 255.0
        
        # Apply colormap
        colored_heatmap = self.apply_colormap(heatmap)
        
        # Blend
        overlay = (
            colored_heatmap * alpha + 
            original_image * (1 - alpha)
        ).astype(np.uint8)
        
        return overlay


class LesionDetector:
    """
    Detect and segment lesions/tumors from heatmaps.
    
    Features:
    - Thresholding-based detection
    - Connected component analysis
    - Size filtering
    - Bounding box extraction
    """
    
    def __init__(
        self,
        detection_threshold: float = 0.5,
        min_lesion_size: int = 100,  # Minimum pixels
        max_lesion_size: Optional[int] = None
    ):
        self.detection_threshold = detection_threshold
        self.min_lesion_size = min_lesion_size
        self.max_lesion_size = max_lesion_size
    
    def detect_lesions(
        self,
        heatmap: np.ndarray,
        return_masks: bool = True
    ) -> List[Dict]:
        """
        Detect lesions from probability heatmap.
        
        Returns:
            List of lesion dictionaries with properties
        """
        # Threshold
        binary_mask = (heatmap > self.detection_threshold).astype(np.uint8)
        
        # Connected components
        labeled, num_features = ndimage.label(binary_mask)
        
        lesions = []
        
        for label_id in range(1, num_features + 1):
            # Extract component
            component_mask = (labeled == label_id)
            area = component_mask.sum()
            
            # Size filtering
            if area < self.min_lesion_size:
                continue
            if self.max_lesion_size and area > self.max_lesion_size:
                continue
            
            # Get bounding box
            coords = np.argwhere(component_mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Calculate statistics
            lesion_probs = heatmap[component_mask]
            avg_confidence = lesion_probs.mean()
            max_confidence = lesion_probs.max()
            
            lesion_info = {
                'id': label_id,
                'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                'area': int(area),
                'avg_confidence': float(avg_confidence),
                'max_confidence': float(max_confidence),
                'center': (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
            }
            
            if return_masks:
                lesion_info['mask'] = component_mask
            
            lesions.append(lesion_info)
        
        # Sort by confidence
        lesions.sort(key=lambda x: x['avg_confidence'], reverse=True)
        
        return lesions
    
    def draw_detections(
        self,
        image: np.ndarray,
        lesions: List[Dict],
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 3,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes around detected lesions.
        
        Returns:
            image_with_boxes: Image with drawn detections
        """
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        
        for lesion in lesions:
            x_min, y_min, x_max, y_max = lesion['bbox']
            
            # Draw rectangle
            draw.rectangle(
                [x_min, y_min, x_max, y_max],
                outline=color,
                width=thickness
            )
            
            # Draw label
            if show_labels:
                label = f"#{lesion['id']}: {lesion['avg_confidence']*100:.1f}%"
                draw.text(
                    (x_min, y_min - 15),
                    label,
                    fill=color
                )
        
        return np.array(img_pil)


class MultiScaleHeatmapAggregator:
    """
    Aggregate heatmaps from multiple scales/magnifications.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int],
        scales: List[float] = [1.0, 0.5, 0.25],
        scale_weights: Optional[List[float]] = None
    ):
        self.image_size = image_size
        self.scales = scales
        
        # Default: equal weights
        if scale_weights is None:
            scale_weights = [1.0 / len(scales)] * len(scales)
        self.scale_weights = scale_weights
        
        # Heatmap generator per scale
        self.generators = [
            HeatmapGenerator(image_size, aggregation_method='weighted_average')
            for _ in scales
        ]
    
    def add_patch_prediction(
        self,
        scale_idx: int,
        x: int,
        y: int,
        probability: float,
        confidence: float
    ):
        """Add prediction for specific scale"""
        self.generators[scale_idx].add_patch_prediction(
            x, y, probability, confidence
        )
    
    def generate_multiscale_heatmap(self) -> np.ndarray:
        """Generate weighted combination of all scales"""
        # Generate individual heatmaps
        scale_heatmaps = [
            gen.generate_heatmap() for gen in self.generators
        ]
        
        # Weighted combination
        combined = np.zeros_like(scale_heatmaps[0])
        for heatmap, weight in zip(scale_heatmaps, self.scale_weights):
            combined += heatmap * weight
        
        # Normalize
        if combined.max() > 0:
            combined = (combined - combined.min()) / (combined.max() - combined.min())
        
        return combined


# ============================================
# UTILITY FUNCTIONS
# ============================================

def calculate_tumor_burden(
    lesions: List[Dict],
    image_size: Tuple[int, int]
) -> Dict:
    """
    Calculate tumor burden metrics.
    """
    total_area = image_size[0] * image_size[1]
    tumor_area = sum(lesion['area'] for lesion in lesions)
    
    return {
        'num_lesions': len(lesions),
        'total_tumor_area_pixels': tumor_area,
        'tumor_burden_percentage': (tumor_area / total_area) * 100,
        'largest_lesion_area': max((l['area'] for l in lesions), default=0),
        'average_lesion_area': tumor_area / len(lesions) if lesions else 0
    }


def create_uncertainty_map(
    predictions: List[Dict],
    positions: List[Tuple[int, int]],
    image_size: Tuple[int, int],
    patch_size: int = 224
) -> np.ndarray:
    """
    Create uncertainty map showing prediction variability.
    Higher values = model is uncertain.
    """
    uncertainty_map = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    count_map = np.zeros((image_size[1], image_size[0]), dtype=np.int32)
    
    for pred, (x, y) in zip(predictions, positions):
        prob = pred['tumor_probability']
        
        # Entropy-based uncertainty: max at 0.5, min at 0 or 1
        entropy = -prob * np.log(prob + 1e-10) - (1-prob) * np.log(1-prob + 1e-10)
        uncertainty = entropy / np.log(2)  # Normalize to [0, 1]
        
        x_end = min(x + patch_size, image_size[0])
        y_end = min(y + patch_size, image_size[1])
        
        uncertainty_map[y:y_end, x:x_end] += uncertainty
        count_map[y:y_end, x:x_end] += 1
    
    # Average
    uncertainty_map = np.divide(
        uncertainty_map,
        count_map,
        out=np.zeros_like(uncertainty_map),
        where=count_map > 0
    )
    
    return uncertainty_map


if __name__ == "__main__":
    print("🗺️  Heatmap Generation and Aggregation Module")
    print("=" * 70)
    
    # Test heatmap generation
    image_size = (2048, 2048)
    generator = HeatmapGenerator(image_size, patch_size=224)
    
    # Simulate patch predictions
    np.random.seed(42)
    num_patches = 50
    
    for i in range(num_patches):
        x = np.random.randint(0, image_size[0] - 224)
        y = np.random.randint(0, image_size[1] - 224)
        prob = np.random.rand()
        conf = np.random.rand() * 100
        
        generator.add_patch_prediction(x, y, prob, conf)
    
    # Generate heatmap
    heatmap = generator.generate_heatmap()
    print(f"✅ Generated heatmap: {heatmap.shape}")
    print(f"   Value range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    # Detect lesions
    detector = LesionDetector(detection_threshold=0.6, min_lesion_size=50)
    lesions = detector.detect_lesions(heatmap)
    
    print(f"\n✅ Detected {len(lesions)} lesions")
    if lesions:
        print(f"   Top lesion: {lesions[0]['avg_confidence']*100:.1f}% confidence")
        print(f"   Area: {lesions[0]['area']} pixels")
    
    print("\n🗺️  Aggregation module ready for use!")
