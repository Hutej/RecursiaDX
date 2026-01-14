# 🎯 Multi-Scale Attention Module
# Attention mechanisms for interpretable histopathology analysis

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism to focus on important regions.
    Generates attention maps showing where the model is looking.
    """
    
    def __init__(self, in_channels: int = 2048, reduction: int = 8):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.bn1 = nn.BatchNorm2d(in_channels // reduction)
        self.conv2 = nn.Conv2d(in_channels // reduction, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature maps
        Returns:
            attention_map: (B, 1, H, W) attention weights
            attended_features: (B, C, H, W) weighted features
        """
        # Generate attention map
        attention = F.relu(self.bn1(self.conv1(x)))
        attention = self.conv2(attention)
        attention_map = self.sigmoid(attention)
        
        # Apply attention
        attended_features = x * attention_map
        
        return attention_map, attended_features


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention that combines features from different magnifications.
    
    Key features:
    - Processes patches at multiple scales (10x, 20x, 40x)
    - Learns which scale is most informative per region
    - Produces interpretable attention weights per scale
    """
    
    def __init__(
        self,
        feature_dims: List[int] = [2048, 2048, 2048],  # Features per scale
        num_scales: int = 3,
        hidden_dim: int = 512,
        attention_heads: int = 8
    ):
        super(MultiScaleAttention, self).__init__()
        
        self.num_scales = num_scales
        self.feature_dims = feature_dims
        self.attention_heads = attention_heads
        
        # Project features from each scale to common dimension
        self.scale_projectors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            for dim in feature_dims
        ])
        
        # Multi-head attention for scale fusion
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=0.2,
            batch_first=True
        )
        
        # Scale importance weighting
        self.scale_weights = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=1)
        )
        
        # Spatial attention per scale
        self.spatial_attentions = nn.ModuleList([
            SpatialAttention(in_channels=dim)
            for dim in feature_dims
        ])
        
    def forward(
        self,
        scale_features: List[torch.Tensor],
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Args:
            scale_features: List of (B, C, H, W) features from each scale
            return_attention: Whether to return attention weights
            
        Returns:
            fused_features: (B, hidden_dim) fused multi-scale features
            attention_dict: Dictionary with attention weights if requested
        """
        batch_size = scale_features[0].size(0)
        
        # Apply spatial attention to each scale
        spatial_attention_maps = []
        attended_features = []
        
        for idx, (features, spatial_attn) in enumerate(zip(scale_features, self.spatial_attentions)):
            attn_map, attn_features = spatial_attn(features)
            spatial_attention_maps.append(attn_map)
            attended_features.append(attn_features)
        
        # Project each scale to common dimension
        projected_features = []
        for idx, (features, projector) in enumerate(zip(attended_features, self.scale_projectors)):
            proj_feat = projector(features)  # (B, hidden_dim)
            projected_features.append(proj_feat)
        
        # Stack for attention: (B, num_scales, hidden_dim)
        stacked_features = torch.stack(projected_features, dim=1)
        
        # Multi-head attention across scales
        attended, scale_attention_weights = self.scale_attention(
            stacked_features,
            stacked_features,
            stacked_features
        )  # (B, num_scales, hidden_dim)
        
        # Calculate scale importance weights
        concat_features = attended.reshape(batch_size, -1)  # (B, num_scales * hidden_dim)
        scale_weights = self.scale_weights(concat_features)  # (B, num_scales)
        
        # Weighted fusion
        scale_weights = scale_weights.unsqueeze(-1)  # (B, num_scales, 1)
        fused_features = torch.sum(attended * scale_weights, dim=1)  # (B, hidden_dim)
        
        # Prepare attention outputs
        attention_dict = None
        if return_attention:
            attention_dict = {
                'scale_weights': scale_weights.squeeze(-1),  # (B, num_scales)
                'spatial_attention': spatial_attention_maps,  # List of (B, 1, H, W)
                'scale_attention_weights': scale_attention_weights  # (B, num_scales, num_scales)
            }
        
        return fused_features, attention_dict


class PatchAttentionAggregator(nn.Module):
    """
    Aggregates predictions from multiple patches using attention.
    Used for whole-slide image classification.
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 2,
        attention_dim: int = 128
    ):
        super(PatchAttentionAggregator, self).__init__()
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(
        self,
        patch_features: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            patch_features: (B, N, feature_dim) features from N patches
            return_attention: Whether to return attention weights
            
        Returns:
            logits: (B, num_classes) classification logits
            attention_weights: (B, N) attention weight per patch
        """
        # Calculate attention weights for each patch
        attention_logits = self.attention(patch_features)  # (B, N, 1)
        attention_weights = F.softmax(attention_logits, dim=1)  # (B, N, 1)
        
        # Weighted aggregation
        aggregated = torch.sum(patch_features * attention_weights, dim=1)  # (B, feature_dim)
        
        # Classification
        logits = self.classifier(aggregated)  # (B, num_classes)
        
        if return_attention:
            return logits, attention_weights.squeeze(-1)
        return logits, None


class GradCAMAttention:
    """
    Grad-CAM based attention for interpretability.
    Shows which regions contribute most to predictions.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM attention map.
        
        Args:
            input_tensor: (1, C, H, W) input image
            target_class: Target class for CAM (None = predicted class)
            
        Returns:
            cam: (H, W) attention map normalized to [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Calculate CAM
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=1).squeeze()  # (H, W)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def __del__(self):
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()


# ============================================
# UTILITY FUNCTIONS
# ============================================

def visualize_attention_map(
    attention_map: np.ndarray,
    original_size: Tuple[int, int],
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Resize and colorize attention map for visualization.
    """
    from PIL import Image
    from matplotlib import cm
    
    # Resize to original size
    attention_pil = Image.fromarray((attention_map * 255).astype(np.uint8))
    attention_resized = attention_pil.resize(original_size, Image.BILINEAR)
    attention_array = np.array(attention_resized) / 255.0
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored_attention = cmap(attention_array)[:, :, :3]
    
    return (colored_attention * 255).astype(np.uint8)


def aggregate_patch_attentions(
    patch_attentions: List[np.ndarray],
    patch_positions: List[Tuple[int, int]],
    image_size: Tuple[int, int],
    aggregation: str = 'max'
) -> np.ndarray:
    """
    Aggregate attention maps from multiple patches into full image heatmap.
    
    Args:
        patch_attentions: List of attention maps per patch
        patch_positions: List of (x, y) positions for each patch
        image_size: (width, height) of full image
        aggregation: 'max', 'mean', or 'weighted'
        
    Returns:
        full_attention: (H, W) aggregated attention map
    """
    height, width = image_size
    attention_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)
    
    for attention, (x, y) in zip(patch_attentions, patch_positions):
        h, w = attention.shape
        
        if aggregation == 'max':
            attention_map[y:y+h, x:x+w] = np.maximum(
                attention_map[y:y+h, x:x+w],
                attention
            )
        else:  # mean or weighted
            attention_map[y:y+h, x:x+w] += attention
            count_map[y:y+h, x:x+w] += 1
    
    if aggregation in ['mean', 'weighted']:
        attention_map = np.divide(
            attention_map,
            count_map,
            out=np.zeros_like(attention_map),
            where=count_map > 0
        )
    
    return attention_map


if __name__ == "__main__":
    print("🎯 Multi-Scale Attention Module")
    print("=" * 70)
    
    # Test multi-scale attention
    batch_size = 4
    scales = 3
    
    # Simulate features from 3 scales
    scale_features = [
        torch.randn(batch_size, 2048, 7, 7),   # Scale 1: 40x
        torch.randn(batch_size, 2048, 7, 7),   # Scale 2: 20x
        torch.randn(batch_size, 2048, 7, 7)    # Scale 3: 10x
    ]
    
    # Initialize attention module
    attention = MultiScaleAttention(
        feature_dims=[2048, 2048, 2048],
        num_scales=3,
        hidden_dim=512,
        attention_heads=8
    )
    
    # Forward pass
    fused_features, attention_dict = attention(scale_features, return_attention=True)
    
    print(f"✅ Fused features shape: {fused_features.shape}")
    print(f"✅ Scale weights shape: {attention_dict['scale_weights'].shape}")
    print(f"✅ Scale weights (batch 0): {attention_dict['scale_weights'][0].detach().numpy()}")
    print("\n🎯 Multi-scale attention ready for use!")
