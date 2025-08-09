#!/usr/bin/env python3
"""
Compatibility layer for the refactored ValidationBatchProcessor.

This module provides backward compatibility by importing the new modular
ValidationBatchProcessor while maintaining the original API.

⚠️ DEPRECATION NOTICE:
The original validation_batch_processor.py exceeded 400 lines and violated SRP.
It has been refactored into focused, modular components under 400 lines each.

New modular structure:
- validation/config_loader.py - Configuration and class mapping
- validation/target_processor.py - Target format conversion
- validation/model_inference.py - Model inference operations  
- validation/metrics_calculator.py - Loss and mAP computation
- validation/batch_processor.py - Main orchestration (260 lines)

Please migrate to the new structure:
from smartcash.model.training.core.validation import ValidationBatchProcessor
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, TypedDict

# Import the new modular components
from .validation.batch_processor import (
    ValidationBatchProcessor as _NewValidationBatchProcessor,
    create_validation_batch_processor,
    BatchMetrics
)

# Re-export for backward compatibility
BatchData = TypedDict('BatchData', {'images': Any, 'boxes': Any, 'labels': Any})


class ValidationBatchProcessor(_NewValidationBatchProcessor):
    """
    Backward compatibility wrapper for the refactored ValidationBatchProcessor.
    
    This class maintains the original API while using the new modular implementation
    under the hood.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with deprecation warning."""
        warnings.warn(
            "The original ValidationBatchProcessor has been refactored into modular components. "
            "Please use: from smartcash.model.training.core.validation import ValidationBatchProcessor",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
    
    # Maintain any legacy method names if needed
    def _calculate_map_metrics(self, *args, **kwargs):
        """Legacy method name compatibility."""
        return self.metrics_calculator.compute_map_metrics(*args, **kwargs)


# Export the factory function for convenience
__all__ = [
    'ValidationBatchProcessor',
    'create_validation_batch_processor', 
    'BatchMetrics',
    'BatchData'
]