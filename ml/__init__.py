"""
RecursiaDx ML Module

A comprehensive machine learning module for tumor detection using ResNet50.
"""

__version__ = "1.0.0"
__author__ = "RecursiaDx Team"

# Import main classes for easy access
from .models.tumor_predictor import TumorPredictor
from .utils.data_manager import DataManager
from .config.config import get_config

__all__ = [
    'TumorPredictor',
    'DataManager', 
    'get_config'
]