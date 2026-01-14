"""
Backbone model loader for feature extraction.
Supports frozen pretrained models for WSI tile embedding.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BackboneLoader:
    """Load and configure pretrained backbone models."""
    
    SUPPORTED_MODELS = {
        'resnet50-imagenet': {
            'feature_dim': 2048,
            'input_size': 224,
            'description': 'ResNet50 pretrained on ImageNet'
        },
        'resnet50-simclr': {
            'feature_dim': 2048,
            'input_size': 224,
            'description': 'ResNet50 with SimCLR self-supervised learning'
        }
    }
    
    @staticmethod
    def load_backbone(
        model_name: str = 'resnet50-imagenet',
        freeze: bool = True,
        device: Optional[torch.device] = None
    ) -> Tuple[nn.Module, int]:
        """
        Load pretrained backbone model.
        
        Args:
            model_name: Name of model to load
            freeze: If True, freeze all parameters (no training)
            device: Device to load model on
        
        Returns:
            Tuple of (model, feature_dim)
        """
        if model_name not in BackboneLoader.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Choose from: {list(BackboneLoader.SUPPORTED_MODELS.keys())}"
            )
        
        logger.info(f"Loading backbone: {model_name}")
        
        if model_name == 'resnet50-imagenet':
            model, feature_dim = BackboneLoader._load_resnet50_imagenet()
        elif model_name == 'resnet50-simclr':
            model, feature_dim = BackboneLoader._load_resnet50_simclr()
        else:
            raise ValueError(f"Model {model_name} not implemented")
        
        # Freeze parameters if requested
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
            logger.info("Model parameters frozen")
        
        # Helper validation
        if not torch.cuda.is_available():
             raise RuntimeError("CUDA GPU REQUIRED. CPU execution is DISABLED.")

        # Move to device (Force CUDA)
        if device is None:
            device = torch.device("cuda")
        elif device.type != 'cuda':
            raise RuntimeError(f"CPU DEVICE DISALLOWED: {device}")
            
        model = model.to(device)
        logger.info(f"Model moved to {device}")
        
        # Set to eval mode
        model.eval()
        
        logger.info(f"Backbone loaded: {model_name} (feature_dim={feature_dim})")
        
        return model, feature_dim
    
    @staticmethod
    def _load_resnet50_imagenet() -> Tuple[nn.Module, int]:
        """
        Load ResNet50 pretrained on ImageNet.
        
        Extracts 2048-dim features after global average pooling.
        Removes classification head by replacing fc with Identity.
        
        Returns:
            Tuple of (model, feature_dim)
        """
        # Load pretrained ResNet50
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove classification head
        # Replace fc layer with Identity to get (B, 2048) embeddings
        model.fc = nn.Identity()
        
        feature_dim = 2048
        
        logger.info(f"Loaded ResNet50-ImageNet (weights: IMAGENET1K_V1)")
        logger.info(f"Feature extraction: After avgpool → fc=Identity → (B, {feature_dim})")
        
        return model, feature_dim
    
    @staticmethod
    def _load_resnet50_simclr() -> Tuple[nn.Module, int]:
        """
        Load ResNet50 with SimCLR self-supervised pretraining.
        
        Note: Requires timm or custom weights.
        Placeholder for future implementation.
        
        Returns:
            Tuple of (model, feature_dim)
        """
        raise NotImplementedError(
            "ResNet50-SimCLR not yet implemented. "
            "Requires timm or custom pretrained weights."
        )
    
    @staticmethod
    def get_model_info(model_name: str) -> dict:
        """
        Get information about a model.
        
        Args:
            model_name: Name of model
        
        Returns:
            Dictionary with model information
        """
        if model_name not in BackboneLoader.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported")
        
        return BackboneLoader.SUPPORTED_MODELS[model_name]
    
    @staticmethod
    def get_input_transforms(model_name: str):
        """
        Get preprocessing transforms for model.
        
        Args:
            model_name: Name of model
        
        Returns:
            torchvision transforms
        """
        from torchvision import transforms
        
        if model_name.startswith('resnet50'):
            # ImageNet normalization
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            raise ValueError(f"Transforms for {model_name} not defined")


def test_backbone_loading():
    """Test backbone loading and feature extraction."""
    import numpy as np
    
    print("Testing backbone loading...")
    
    # Load model
    if not torch.cuda.is_available():
        raise RuntimeError("Test failed: CUDA required")
        
    device = torch.device('cuda')
    model, feature_dim = BackboneLoader.load_backbone(
        model_name='resnet50-imagenet',
        freeze=True,
        device=device
    )
    
    # Create dummy input (B=2, C=3, H=256, W=256)
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    
    # Get transforms
    transforms = BackboneLoader.get_input_transforms('resnet50-imagenet')
    
    # Forward pass
    with torch.no_grad():
        features = model(dummy_input)
    
    print(f"\n✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output shape: {features.shape}")
    print(f"✓ Expected: (2, {feature_dim})")
    print(f"✓ Feature dim: {feature_dim}")
    
    assert features.shape == (2, feature_dim), "Feature shape mismatch!"
    assert not features.requires_grad, "Model should be frozen!"
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_backbone_loading()
