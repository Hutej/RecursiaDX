# Gigapixel Histopathology Analysis Pipeline
# Multi-scale tiling and attention-based lesion detection

__version__ = "1.0.0"
__author__ = "Histopathology AI Team"

from .tiling import GigapixelTiler, PatchExtractor
from .attention import MultiScaleAttention, SpatialAttention
from .classifier import PatchClassifier, EnsembleClassifier
from .aggregation import HeatmapGenerator, LesionDetector
from .pipeline import HistopathologyPipeline

__all__ = [
    'GigapixelTiler',
    'PatchExtractor',
    'MultiScaleAttention',
    'SpatialAttention',
    'PatchClassifier',
    'EnsembleClassifier',
    'HeatmapGenerator',
    'LesionDetector',
    'HistopathologyPipeline'
]
