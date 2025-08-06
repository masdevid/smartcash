"""
Early stopping module for SmartCash training optimization.

This module provides a modular early stopping system with different strategies
for various training scenarios.
"""

from .base import BaseEarlyStopping
from .standard import StandardEarlyStopping
from .multi_metric import MultiMetricEarlyStopping
from .adaptive import AdaptiveEarlyStopping
from .phase_specific import PhaseSpecificEarlyStopping
from .factory import (
    create_early_stopping,
    create_adaptive_early_stopping,
    create_phase_specific_early_stopping
)

# For backward compatibility
EarlyStopping = StandardEarlyStopping

__all__ = [
    'BaseEarlyStopping',
    'StandardEarlyStopping',
    'MultiMetricEarlyStopping', 
    'AdaptiveEarlyStopping',
    'PhaseSpecificEarlyStopping',
    'EarlyStopping',  # Backward compatibility alias
    'create_early_stopping',
    'create_adaptive_early_stopping',
    'create_phase_specific_early_stopping'
]