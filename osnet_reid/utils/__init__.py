"""
Utility modules for OSNet ReID.

Subpackages:
- data: Dataset loading, transforms, sampler
- callbacks: LR scheduler, early stopping, EMA
"""

# Data utilities
from .data import (
    ReIDDataset,
    RandomIdentitySampler,
    get_train_transforms, get_val_transforms,
)

# Callbacks
from .callbacks import ReduceLROnPlateau, EarlyStopping, ModelEMA

# General utilities
from .general import init_seeds, colorstr, increment_path

__all__ = [
    # Data
    'ReIDDataset', 'RandomIdentitySampler',
    'get_train_transforms', 'get_val_transforms',
    # Callbacks
    'ReduceLROnPlateau', 'EarlyStopping', 'ModelEMA',
    # General
    'init_seeds', 'colorstr', 'increment_path',
]
