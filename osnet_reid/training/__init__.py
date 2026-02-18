"""
Training module for OSNet ReID model.

Provides:
- Training configuration and OSNet variant registry
- Training pipeline (CE + Triplet joint training)
- Validation and evaluation utilities
"""
from .config import (
    AVAILABLE_MODELS,
    list_available_models,
    validate_model,
    build_config,
    load_yaml,
    get_model_info,
    get_all_model_names,
)

from .trainer import Trainer

from .evaluator import (
    validate,
    find_best_threshold,
)

__all__ = [
    # Configuration
    'AVAILABLE_MODELS',
    'list_available_models',
    'validate_model',
    'build_config',
    'load_yaml',
    'get_model_info',
    'get_all_model_names',
    # Training
    'Trainer',
    # Evaluation
    'validate',
    'find_best_threshold',
]
