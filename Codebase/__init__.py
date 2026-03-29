"""
Food Freshness Detection - Complete Package
Custom 11-layer CNN for binary food classification
"""

from .model import FoodFreshnessDetectionCNN, create_model
from .utils import (
    FoodFreshnessDataset,
    create_dataloaders,
    get_transforms,
    calculate_metrics,
    visualize_batch
)
from .analyze import FoodAnalyzer
from .train import Trainer

__version__ = '1.0.0'
__all__ = [
    'FoodFreshnessDetectionCNN',
    'create_model',
    'FoodFreshnessDataset',
    'create_dataloaders',
    'get_transforms',
    'calculate_metrics',
    'visualize_batch',
    'FoodAnalyzer',
    'Trainer'
]
