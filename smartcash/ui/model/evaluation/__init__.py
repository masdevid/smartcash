"""
Evaluation Module - Comprehensive model evaluation across research scenarios

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/model/evaluation/__init__.py
"""

from .evaluation_uimodule import EvaluationUIModule
from .evaluation_ui_factory import EvaluationUIFactory, create_evaluation_display

def initialize_evaluation_ui(config=None, **kwargs):
    """
    Initialize and display the evaluation UI.
    
    Args:
        config: Optional configuration dict
        **kwargs: Additional arguments for UI initialization
        
    Returns:
        None (displays the UI using IPython.display)
    """
    EvaluationUIFactory.create_and_display_evaluation(config=config, **kwargs)

# Export main classes and functions
__all__ = [
    'EvaluationUIModule',
    'EvaluationUIFactory',
    'initialize_evaluation_ui',
    'create_evaluation_display'
]