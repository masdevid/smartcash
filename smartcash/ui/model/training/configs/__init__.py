"""
File: smartcash/ui/model/training/configs/__init__.py
Training configuration management - Updated for unified training pipeline.
"""

# Unified training configuration (primary)
from .unified_training_defaults import (
    get_unified_training_defaults,
    get_backbone_options,
    get_training_mode_options,
    get_loss_type_options,
    get_early_stopping_metric_options,
    get_early_stopping_mode_options,
    get_single_phase_layer_mode_options,
    validate_unified_training_config
)

# Legacy configuration removed - use unified system instead

from .training_config_handler import TrainingConfigHandler

__all__ = [
    # Unified training configuration (primary)
    'get_unified_training_defaults',
    'get_backbone_options',
    'get_training_mode_options', 
    'get_loss_type_options',
    'get_early_stopping_metric_options',
    'get_early_stopping_mode_options',
    'get_single_phase_layer_mode_options',
    'validate_unified_training_config',
    
    'TrainingConfigHandler'
]