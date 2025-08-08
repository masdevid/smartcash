"""
Visualization package for SmartCash training metrics and results.

This package provides tools for visualizing training metrics, generating charts,
and creating comprehensive dashboards for model analysis and evaluation.

Main components:
- VisualizationManager: Central class for managing all visualizations
- Charts: Subpackage for generating different types of charts
- Utils: Helper functions and validation utilities
- Types: Type hints and data classes for type safety

Example usage:
    ```python
    from smartcash.model.training.visualization import VisualizationManager
    
    # Initialize the visualization manager
    viz = VisualizationManager(
        num_classes_per_layer={'layer1': 10, 'layer2': 5},
        class_names={'layer1': ['class1', 'class2', ...], ...},
        save_dir='visualizations',
        verbose=True
    )
    
    # Update metrics during training
    viz.update_metrics(
        epoch=1,
        metrics={'loss': 0.5, 'accuracy': 0.8},
        phase='train',
        learning_rate=0.001
    )
    
    # Generate visualizations
    viz.generate_training_curves()
    viz.generate_confusion_matrices()
    viz.generate_research_dashboard()
    ```
"""

# Import main components
from .manager import VisualizationManager
from . import charts
from . import utils
from . import types

# Re-export commonly used components
__all__ = [
    'VisualizationManager',
    'charts',
    'utils',
    'types'
]

# Package version
__version__ = '0.1.0'
