"""
File: smartcash/ui_components/__init__.py
Author: Alfrida Sabar
Deskripsi: Package initialization untuk UI components.
"""

# Data components
from smartcash.ui_components.data_components import (
    create_dataset_info_ui,
    create_split_dataset_ui,
    create_data_utils_ui,
    create_data_handling_ui
)


# Dataset components
from smartcash.ui_components.dataset_components import (
    create_dataset_ui
)

# Directory components
from smartcash.ui_components.directory_components import (
    create_directory_ui
)

# Augmentation components
from smartcash.ui_components.augmentation_components import (
    create_augmentation_controls,
    create_augmentation_buttons,
    create_augmentation_ui
)

# Config components
from smartcash.ui_components.config_components import (
    create_global_config_ui,
    create_pipeline_config_ui
)

# Model components
from smartcash.ui_components.model_components import (
    create_model_initialization_ui,
    create_model_visualizer_ui,
    create_checkpoint_manager_ui,
    create_model_optimization_ui,
    create_model_exporter_ui,
    create_model_manager_ui
)

# Model playground components
from smartcash.ui_components.model_playground_components import (
    create_model_selector_controls,
    create_model_option_controls,
    create_model_test_controls,
    create_model_playground_ui
)

# Evaluation components
from smartcash.ui_components.evaluation_components import (
    create_model_selector_controls as create_eval_model_selector_controls,
    create_evaluation_settings,
    create_evaluation_ui
)

# Research components
from smartcash.ui_components.research_components import (
    create_scenario_checkboxes,
    create_evaluation_controls,
    create_research_ui
)

# Training components
from smartcash.ui_components.training_components import (
    create_training_controls,
    create_drive_backup_control,
    create_status_display,
    create_visualization_tabs,
    create_training_ui,
    create_training_pipeline_ui,
    create_training_config_ui
)

# Repository components
from smartcash.ui_components.repository_components import (
    create_repository_ui
)

__all__ = [
    # Data
    'create_dataset_info_ui', 'create_split_dataset_ui', 'create_data_utils_ui',
    'create_data_handling_ui',
    
    # Dataset
    'create_dataset_ui',
    
    # Directory
    'create_directory_ui',
    
    # Augmentation
    'create_augmentation_controls', 'create_augmentation_buttons', 'create_augmentation_ui',
    
    # Config
    'create_global_config_ui', 'create_pipeline_config_ui',
    
    # Model
    'create_model_initialization_ui', 'create_model_visualizer_ui', 'create_checkpoint_manager_ui',
    'create_model_optimization_ui', 'create_model_exporter_ui', 'create_model_manager_ui',
    
    # Model playground
    'create_model_selector_controls', 'create_model_option_controls',
    'create_model_test_controls', 'create_model_playground_ui',
    
    # Evaluation
    'create_eval_model_selector_controls', 'create_evaluation_settings', 'create_evaluation_ui',
    
    # Research
    'create_scenario_checkboxes', 'create_evaluation_controls', 'create_research_ui',
    
    # Training
    'create_training_controls', 'create_drive_backup_control', 'create_status_display',
    'create_visualization_tabs', 'create_training_ui', 'create_training_pipeline_ui',
    'create_training_config_ui',
    
    # Repository
    'create_repository_ui'
]