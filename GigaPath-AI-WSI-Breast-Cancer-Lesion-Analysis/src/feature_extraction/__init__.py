"""
Feature extraction module for WSI tiles.
Extracts deep learning embeddings using frozen pretrained backbones.
"""
from .backbones import BackboneLoader
from .feature_extractor import FeatureExtractor

__all__ = [
    'BackboneLoader',
    'FeatureExtractor',
]
