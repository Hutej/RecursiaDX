"""
Explainability module for MIL model interpretation.
"""
from .heatmap_generator import HeatmapGenerator
from .gradcam import GradCAM, GradCAMAggregator

__all__ = [
    'HeatmapGenerator',
    'GradCAM',
    'GradCAMAggregator',
]
