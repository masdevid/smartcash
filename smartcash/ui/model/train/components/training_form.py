"""
File: smartcash/ui/model/train/components/training_form.py
Description: Training configuration form component for the training UI.
"""

import ipywidgets as widgets
from typing import Dict, Any, Callable
from smartcash.ui.logger import get_module_logger
from ..configs.train_defaults import get_layer_mode_configs, get_optimization_types


def create_training_form(training_config: Dict[str, Any], ui_config: Dict[str, Any]) -> widgets.Widget:
    """Create training configuration form.
    
    Args:
        training_config: Dictionary containing training configuration
        ui_config: UI configuration parameters
        
    Returns:
        Widget containing the training form
    """
    logger = get_module_logger("smartcash.ui.model.train.components")
    
    try:
        layer_configs = get_layer_mode_configs()
        optimization_types = get_optimization_types()
        
        # Layer mode selection
        layer_mode_dropdown = widgets.Dropdown(
            description="Training Mode:",
            options=[(config['display_name'], key) for key, config in layer_configs.items()],
            value=training_config.get('layer_mode', 'single'),
            style={'description_width': '150px'},
            layout=widgets.Layout(width='100%')
        )
        
        # Basic training parameters
        epochs_input = widgets.IntSlider(
            description="Epochs:",
            value=training_config.get('epochs', 100),
            min=1,
            max=1000,
            step=10,
            style={'description_width': '150px'},
            layout=widgets.Layout(width='100%')
        )
        
        batch_size_input = widgets.IntSlider(
            description="Batch Size:",
            value=training_config.get('batch_size', 16),
            min=1,
            max=256,
            step=2,
            style={'description_width': '150px'},
            layout=widgets.Layout(width='100%')
        )
        
        learning_rate_input = widgets.FloatLogSlider(
            description="Learning Rate:",
            value=training_config.get('learning_rate', 0.001),
            base=10,
            min=-6,
            max=0,
            step=0.1,
            style={'description_width': '150px'},
            layout=widgets.Layout(width='100%')
        )
        
        # Optimization type
        optimization_dropdown = widgets.Dropdown(
            description="Optimization:",
            options=[(config['display_name'], key) for key, config in optimization_types.items()],
            value=training_config.get('optimization_type', 'default'),
            style={'description_width': '150px'},
            layout=widgets.Layout(width='100%')
        )
        
        # Advanced options (simplified)
        early_stopping_config = training_config.get('early_stopping', {})
        mixed_precision_checkbox = widgets.Checkbox(
            description="Mixed Precision",
            value=training_config.get('mixed_precision', True),
            style={'description_width': '150px'}
        )
        
        early_stopping_checkbox = widgets.Checkbox(
            description="Early Stopping",
            value=early_stopping_config.get('enabled', True),
            style={'description_width': '150px'}
        )
        
        # Create form container using VBox
        form_widgets = [
            widgets.HTML("<h4>🚀 Training Configuration</h4>"),
            layer_mode_dropdown,
            epochs_input,
            batch_size_input,
            learning_rate_input,
            optimization_dropdown,
            widgets.HTML("<h5>Advanced Options</h5>"),
            widgets.HBox([mixed_precision_checkbox, early_stopping_checkbox])
        ]
        
        form_container = widgets.VBox(
            form_widgets,
            layout=widgets.Layout(
                border='1px solid #ddd',
                padding='15px',
                margin='10px 0',
                border_radius='5px'
            )
        )
        
        # Store references for value retrieval
        form_container._layer_mode_dropdown = layer_mode_dropdown
        form_container._epochs_input = epochs_input
        form_container._batch_size_input = batch_size_input
        form_container._learning_rate_input = learning_rate_input
        form_container._optimization_dropdown = optimization_dropdown
        form_container._mixed_precision_checkbox = mixed_precision_checkbox
        form_container._early_stopping_checkbox = early_stopping_checkbox
        
        # Add method to get form values
        def get_form_values():
            values = {
                'layer_mode': layer_mode_dropdown.value,
                'epochs': int(epochs_input.value),
                'batch_size': int(batch_size_input.value),
                'learning_rate': float(learning_rate_input.value),
                'optimization_type': optimization_dropdown.value,
                'mixed_precision': mixed_precision_checkbox.value,
                'early_stopping_enabled': early_stopping_checkbox.value
            }
            return values
        
        form_container.get_form_values = get_form_values
        
        logger.debug("✅ Training form created successfully")
        return form_container
        
    except Exception as e:
        logger.error(f"Failed to create training form: {e}")
        raise


# For backward compatibility
_create_training_form = create_training_form
