"""
Utility functions for visualization components.

This package provides various utility functions that support the visualization
functionality, including data processing, validation, and formatting.
"""

from .helpers import (
    get_default_layer_metrics,
    initialize_layer_metrics,
    normalize_confusion_matrix,
    ensure_directory_exists,
    format_metric_name,
    safe_divide
)

from .validation import (
    validate_metrics_dict,
    validate_confusion_matrix,
    validate_layer_config,
    validate_learning_rates,
    validate_epoch_metrics,
    validate_visualization_available
)

__all__ = [
    # Helpers
    'get_default_layer_metrics',
    'initialize_layer_metrics',
    'normalize_confusion_matrix',
    'ensure_directory_exists',
    'format_metric_name',
    'safe_divide',
    
    # Validation
    'validate_metrics_dict',
    'validate_confusion_matrix',
    'validate_layer_config',
    'validate_learning_rates',
    'validate_epoch_metrics',
    'validate_visualization_available'
]
