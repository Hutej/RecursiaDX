"""
Gradient-weighted Class Activation Mapping (Grad-CAM) for WSI explainability.

Provides pixel-level visual explanations by computing gradients on CNN backbone.
Used for high-quality clinical/demo visualizations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import cv2

from ..utils.logger import get_logger

logger = get_logger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for visual explainability.
    
    Computes pixel-level heatmaps showing which regions of a tile
    the CNN feature extractor focuses on for classification.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: torch.device
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Feature extraction model (e.g., ResNet)
            target_layer: Layer to extract gradients from (e.g., layer4)
            device: CUDA device
        """
        if device.type != 'cuda':
            raise ValueError(f"Grad-CAM requires CUDA device, got {device.type}")
        
        self.model = model
        self.target_layer = target_layer
        self.device = device
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
        
        logger.debug(f"Grad-CAM initialized on {device} with target layer: {target_layer.__class__.__name__}")
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            """Save activations during forward pass."""
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            """Save gradients during backward pass."""
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
        tile_size: Tuple[int, int] = (256, 256)
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for input tensor.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for gradients (0=benign, 1=malignant)
            tile_size: Size to resize CAM to
        
        Returns:
            CAM heatmap (H, W) normalized to [0, 1]
        """
        self.model.eval()
        
        # Ensure tensor requires grad
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get score for target class
        if output.dim() == 2:  # (batch_size, num_classes)
            class_score = output[0, target_class]
        else:  # (batch_size,) - single output
            class_score = output[0]
        
        # Backward pass
        self.model.zero_grad()
        class_score.backward()
        
        # Compute CAM
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling of gradients to get weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W)
        
        # Apply ReLU (only positive influence)
        cam = F.relu(cam)
        
        # Convert to numpy
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Resize to tile size
        cam = cv2.resize(cam, tile_size, interpolation=cv2.INTER_LINEAR)
        
        logger.debug(f"Generated CAM: shape={cam.shape}, min={cam.min():.4f}, max={cam.max():.4f}")
        
        return cam
    
    def generate_cam_batch(
        self,
        input_tensors: torch.Tensor,
        target_class: int = 1,
        tile_size: Tuple[int, int] = (256, 256)
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmaps for batch of inputs.
        
        Args:
            input_tensors: Batch of input tensors (B, C, H, W)
            target_class: Target class for gradients
            tile_size: Size to resize CAMs to
        
        Returns:
            Array of CAM heatmaps (B, H, W)
        """
        batch_cams = []
        
        for i in range(input_tensors.shape[0]):
            cam = self.generate_cam(
                input_tensors[i:i+1],
                target_class=target_class,
                tile_size=tile_size
            )
            batch_cams.append(cam)
        
        return np.array(batch_cams)
    
    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        # Hooks are automatically removed when target_layer is deleted
        pass


class GradCAMAggregator:
    """
    Aggregate tile-level Grad-CAM heatmaps into full-slide visualization.
    
    Combines pixel-level CAMs with MIL attention weights for dense,
    attention-weighted slide-level heatmaps.
    """
    
    def __init__(
        self,
        tile_size: int = 256,
        slide_dimensions: Tuple[int, int] = (2048, 2048),
        overlap_method: str = 'max'
    ):
        """
        Initialize aggregator.
        
        Args:
            tile_size: Size of individual tiles
            slide_dimensions: Target slide canvas size (W, H)
            overlap_method: How to handle overlaps ('max', 'average', 'weighted_avg')
        """
        self.tile_size = tile_size
        self.slide_dimensions = slide_dimensions
        self.overlap_method = overlap_method
        
        logger.debug(
            f"GradCAM Aggregator initialized: "
            f"tile_size={tile_size}, canvas={slide_dimensions}, "
            f"overlap_method={overlap_method}"
        )
    
    def aggregate(
        self,
        tile_cams: np.ndarray,
        tile_coords: np.ndarray,
        attention_weights: np.ndarray,
        downsample_factor: int = 32
    ) -> np.ndarray:
        """
        Stitch tile Grad-CAMs into full-slide heatmap.
        
        Args:
            tile_cams: Tile-level CAM heatmaps (N, H, W)
            tile_coords: Tile coordinates in level-0 space (N, 2)
            attention_weights: MIL attention scores (N,)
            downsample_factor: Downsample factor from level-0 to canvas
        
        Returns:
            Full-slide Grad-CAM heatmap (H, W) weighted by attention
        """
        logger.info(f"Aggregating {len(tile_cams)} tile Grad-CAMs into slide canvas")
        
        # Create canvas
        canvas_h, canvas_w = self.slide_dimensions
        heatmap_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        weight_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        
        # Normalize attention weights
        attention_norm = attention_weights / (attention_weights.max() + 1e-10)
        
        for i, (cam, coord, attn) in enumerate(zip(tile_cams, tile_coords, attention_norm)):
            # Convert level-0 coordinates to canvas coordinates
            canvas_x = int(coord[0] / downsample_factor)
            canvas_y = int(coord[1] / downsample_factor)
            
            # Calculate canvas tile size
            canvas_tile_size = self.tile_size // downsample_factor
            
            # Resize CAM to canvas tile size
            cam_resized = cv2.resize(
                cam,
                (canvas_tile_size, canvas_tile_size),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Weight CAM by attention score
            weighted_cam = cam_resized * attn
            
            # Calculate placement bounds
            x_end = min(canvas_x + canvas_tile_size, canvas_w)
            y_end = min(canvas_y + canvas_tile_size, canvas_h)
            
            # Skip if out of bounds
            if canvas_x >= canvas_w or canvas_y >= canvas_h:
                continue
            
            # Calculate actual size to place
            place_w = x_end - canvas_x
            place_h = y_end - canvas_y
            
            # Place on canvas
            if self.overlap_method == 'max':
                heatmap_canvas[canvas_y:y_end, canvas_x:x_end] = np.maximum(
                    heatmap_canvas[canvas_y:y_end, canvas_x:x_end],
                    weighted_cam[:place_h, :place_w]
                )
            elif self.overlap_method in ['average', 'weighted_avg']:
                heatmap_canvas[canvas_y:y_end, canvas_x:x_end] += weighted_cam[:place_h, :place_w]
                weight_canvas[canvas_y:y_end, canvas_x:x_end] += 1.0
        
        # Normalize if using averaging
        if self.overlap_method in ['average', 'weighted_avg']:
            mask = weight_canvas > 0
            heatmap_canvas[mask] /= weight_canvas[mask]
        
        # Final normalization to [0, 1]
        if heatmap_canvas.max() > 0:
            heatmap_canvas = (heatmap_canvas - heatmap_canvas.min()) / \
                           (heatmap_canvas.max() - heatmap_canvas.min())
        
        logger.info(
            f"Aggregation complete: canvas shape={heatmap_canvas.shape}, "
            f"min={heatmap_canvas.min():.4f}, max={heatmap_canvas.max():.4f}"
        )
        
        return heatmap_canvas
    
    def apply_colormap(
        self,
        heatmap: np.ndarray,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Apply colormap to heatmap.
        
        Args:
            heatmap: Grayscale heatmap (H, W) in [0, 1]
            colormap: OpenCV colormap
        
        Returns:
            Colored heatmap (H, W, 3) in RGB
        """
        # Convert to uint8
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(heatmap_uint8, colormap)
        
        # Convert BGR to RGB
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        return colored_rgb
