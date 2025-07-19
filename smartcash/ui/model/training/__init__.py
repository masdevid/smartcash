"""
File: smartcash/ui/model/training/__init__.py
Training module package for SmartCash UI using BaseUIModule pattern.
"""

from .training_uimodule import (
    TrainingUIModule,
    create_training_uimodule,
    get_training_uimodule,
    reset_training_uimodule,
    initialize_training_ui
)

from .configs.training_config_handler import TrainingConfigHandler
from .configs.training_defaults import (
    get_default_training_config,
    get_available_optimizers,
    get_available_schedulers,
    TrainingPhase
)

from .operations.training_factory import (
    TrainingOperationFactory,
    TrainingOperationType,
    create_start_training_handler,
    create_stop_training_handler,
    create_resume_training_handler,
    create_validate_training_handler
)

from .components import (
    create_training_ui,
    update_training_ui_from_config,
    get_training_form_values,
    update_metrics_display
)

__all__ = [
    # Main module classes
    'TrainingUIModule',
    'create_training_uimodule',
    'get_training_uimodule', 
    'reset_training_uimodule',
    'initialize_training_ui',
    
    # Configuration
    'TrainingConfigHandler',
    'get_default_training_config',
    'get_available_optimizers',
    'get_available_schedulers',
    'TrainingPhase',
    
    # Operations
    'TrainingOperationFactory',
    'TrainingOperationType',
    'create_start_training_handler',
    'create_stop_training_handler',
    'create_resume_training_handler',
    'create_validate_training_handler',
    
    # UI Components
    'create_training_ui',
    'update_training_ui_from_config',
    'get_training_form_values',
    'update_metrics_display'
]

# Version info
__version__ = "1.0.0"
__author__ = "SmartCash Development Team"
__description__ = "Training module for SmartCash object detection model training with BaseUIModule pattern"