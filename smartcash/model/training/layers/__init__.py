"""
Layer management and validation utilities for the training pipeline.

This package provides functionality for managing and validating model layers,
including layer configuration, validation, and metrics calculation.
"""

# Import core functionality to make it available at package level
from .constants import LAYER_CONFIG, DETECTION_LAYERS, LAYER_NAMES
from .validation import (
    validate_layer_params,
    is_valid_layer_combination,
    auto_determine_layer_mode,
    get_layer_info,
    get_layer_offsets,
    get_num_classes_for_layers,
    get_primary_layer,
    get_total_classes,
    is_multilayer_capable,
    get_valid_layers_only
)

# Re-export for backward compatibility
__all__ = [
    # Constants
    'LAYER_CONFIG',
    'DETECTION_LAYERS',
    'LAYER_NAMES',
    
    # Validation functions
    'validate_layer_params',
    'is_valid_layer_combination',
    'auto_determine_layer_mode',
    
    # Layer information
    'get_layer_info',
    'get_layer_offsets',
    'get_num_classes_for_layers',
    
    # Utility functions
    'get_primary_layer',
    'get_total_classes',
    'is_multilayer_capable',
    'get_valid_layers_only'
]
