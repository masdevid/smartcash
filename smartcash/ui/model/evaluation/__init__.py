"""
Evaluation Module - New Core Pattern
Comprehensive model evaluation across 2×4 research scenarios (2 scenarios × 4 models = 8 tests)

This module provides comprehensive model evaluation capabilities with:
- Position variation scenario testing
- Lighting variation scenario testing
- 2 backbone types × 2 layer modes = 4 model combinations
- Total: 8 evaluation tests with complete metrics analysis
"""

from .evaluation_uimodule import EvaluationUIModule
from .components.evaluation_ui import create_evaluation_ui
from .constants import (
    UI_CONFIG,
    RESEARCH_SCENARIOS,
    MODEL_COMBINATIONS,
    EVALUATION_MATRIX,
    EVALUATION_METRICS
)

__version__ = "2.0.0"
__author__ = "SmartCash Team"

# Export main classes and functions
__all__ = [
    'EvaluationUIModule',
    'UI_CONFIG',
    'RESEARCH_SCENARIOS',
    'MODEL_COMBINATIONS',
    'EVALUATION_MATRIX',
    'EVALUATION_METRICS',
    'create_evaluation_module'
]

def create_evaluation_module(config=None):
    """
    Create and initialize evaluation UI module.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Initialized EvaluationUIModule instance
    """
    module = EvaluationUIModule()
    module.initialize(config)
    return module