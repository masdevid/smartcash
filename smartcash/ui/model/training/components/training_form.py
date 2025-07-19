"""
File: smartcash/ui/model/training/components/training_form.py
Description: Training configuration form component for the training UI.
"""

import ipywidgets as widgets
from typing import Dict, Any, Callable, Optional
from smartcash.ui.logger import get_module_logger
from ..configs.training_defaults import (
    get_available_optimizers, 
    get_available_schedulers,
    TRAINING_VALIDATION_CONFIG
)


def create_training_form(training_config: Dict[str, Any], ui_config: Dict[str, Any]) -> widgets.Widget:
    """Create training configuration form.
    
    Args:
        training_config: Dictionary containing training configuration
        ui_config: UI configuration parameters
        
    Returns:
        Widget containing the training form
    """
    logger = get_module_logger("smartcash.ui.model.training.components")
    
    try:
        optimizers = get_available_optimizers()
        schedulers = get_available_schedulers()
        validation_config = TRAINING_VALIDATION_CONFIG
        
        # Model Selection Section
        model_selection = training_config.get('model_selection', {})
        
        model_source_dropdown = widgets.Dropdown(
            description="Model Source:",
            options=[
                ('From Backbone', 'backbone'),
                ('From Checkpoint', 'checkpoint'),
                ('Pretrained Model', 'pretrained')
            ],
            value=model_selection.get('source', 'backbone'),
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        checkpoint_path_text = widgets.Text(
            description="Checkpoint Path:",
            value=model_selection.get('checkpoint_path', ''),
            placeholder="Path to checkpoint file (only for checkpoint source)",
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        # Basic Training Parameters
        epochs_input = widgets.IntSlider(
            description="Epochs:",
            value=training_config.get('epochs', 100),
            min=validation_config['min_epochs'],
            max=validation_config['max_epochs'],
            step=10,
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        batch_size_input = widgets.IntSlider(
            description="Batch Size:",
            value=training_config.get('batch_size', 16),
            min=validation_config['min_batch_size'],
            max=validation_config['max_batch_size'],
            step=1,
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        learning_rate_input = widgets.FloatLogSlider(
            description="Learning Rate:",
            value=training_config.get('learning_rate', 0.001),
            base=10,
            min=-6,
            max=0,
            step=0.1,
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        # Optimizer and Scheduler
        optimizer_dropdown = widgets.Dropdown(
            description="Optimizer:",
            options=[(name, key) for key, name in optimizers.items()],
            value=training_config.get('optimizer', 'adam'),
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        scheduler_dropdown = widgets.Dropdown(
            description="LR Scheduler:",
            options=[(name, key) for key, name in schedulers.items()],
            value=training_config.get('scheduler', 'cosine'),
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        weight_decay_input = widgets.FloatSlider(
            description="Weight Decay:",
            value=training_config.get('weight_decay', 0.0005),
            min=validation_config['min_weight_decay'],
            max=validation_config['max_weight_decay'],
            step=0.0001,
            readout_format='.4f',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        # Advanced Training Options
        early_stopping_config = training_config.get('early_stopping', {})
        
        mixed_precision_checkbox = widgets.Checkbox(
            description="Mixed Precision Training",
            value=training_config.get('mixed_precision', True),
            style={'description_width': '200px'}
        )
        
        early_stopping_checkbox = widgets.Checkbox(
            description="Early Stopping",
            value=early_stopping_config.get('enabled', True),
            style={'description_width': '200px'}
        )
        
        early_stopping_patience = widgets.IntSlider(
            description="ES Patience:",
            value=early_stopping_config.get('patience', 15),
            min=5,
            max=50,
            step=5,
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        # Data Configuration
        data_config = training_config.get('data', {})
        
        val_split_slider = widgets.FloatSlider(
            description="Validation Split:",
            value=data_config.get('val_split', 0.2),
            min=0.1,
            max=0.5,
            step=0.05,
            readout_format='.2f',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        workers_slider = widgets.IntSlider(
            description="Data Workers:",
            value=data_config.get('workers', 4),
            min=1,
            max=16,
            step=1,
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        # Monitoring Configuration
        monitoring_config = training_config.get('monitoring', {})
        
        save_period_slider = widgets.IntSlider(
            description="Save Period:",
            value=training_config.get('save_period', 10),
            min=1,
            max=50,
            step=1,
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        # UI Configuration
        ui_options = training_config.get('ui', {})
        
        show_advanced_checkbox = widgets.Checkbox(
            description="Show Advanced Options",
            value=ui_options.get('show_advanced_options', False),
            style={'description_width': '200px'}
        )
        
        enable_charts_checkbox = widgets.Checkbox(
            description="Enable Live Charts",
            value=ui_options.get('enable_live_charts', True),
            style={'description_width': '200px'}
        )
        
        # Create form sections
        model_section = widgets.VBox([
            widgets.HTML("<h4>🏗️ Model Configuration</h4>"),
            model_source_dropdown,
            checkpoint_path_text
        ])
        
        training_section = widgets.VBox([
            widgets.HTML("<h4>🚀 Training Parameters</h4>"),
            epochs_input,
            batch_size_input,
            learning_rate_input,
            optimizer_dropdown,
            scheduler_dropdown,
            weight_decay_input
        ])
        
        advanced_section = widgets.VBox([
            widgets.HTML("<h4>⚙️ Advanced Options</h4>"),
            mixed_precision_checkbox,
            early_stopping_checkbox,
            early_stopping_patience,
            save_period_slider
        ])
        
        data_section = widgets.VBox([
            widgets.HTML("<h4>📊 Data Configuration</h4>"),
            val_split_slider,
            workers_slider
        ])
        
        ui_section = widgets.VBox([
            widgets.HTML("<h4>🎨 UI Options</h4>"),
            show_advanced_checkbox,
            enable_charts_checkbox
        ])
        
        # Create main form with accordion
        sections = [
            model_section,
            training_section,
            data_section,
            advanced_section,
            ui_section
        ]
        
        form_accordion = widgets.Accordion(children=sections)
        form_accordion.set_title(0, "🏗️ Model Configuration")
        form_accordion.set_title(1, "🚀 Training Parameters")
        form_accordion.set_title(2, "📊 Data Configuration")
        form_accordion.set_title(3, "⚙️ Advanced Options")
        form_accordion.set_title(4, "🎨 UI Options")
        
        # Set initial accordion state
        form_accordion.selected_index = 1  # Open training parameters by default
        
        # Store widget references for form value extraction
        form_widgets = {
            'model_source': model_source_dropdown,
            'checkpoint_path': checkpoint_path_text,
            'epochs': epochs_input,
            'batch_size': batch_size_input,
            'learning_rate': learning_rate_input,
            'optimizer': optimizer_dropdown,
            'scheduler': scheduler_dropdown,
            'weight_decay': weight_decay_input,
            'mixed_precision': mixed_precision_checkbox,
            'early_stopping': early_stopping_checkbox,
            'early_stopping_patience': early_stopping_patience,
            'val_split': val_split_slider,
            'workers': workers_slider,
            'save_period': save_period_slider,
            'show_advanced': show_advanced_checkbox,
            'enable_charts': enable_charts_checkbox
        }
        
        # Add method to get form values
        def get_form_values() -> Dict[str, Any]:
            """Extract current form values."""
            try:
                return {
                    'model_selection': {
                        'source': form_widgets['model_source'].value,
                        'checkpoint_path': form_widgets['checkpoint_path'].value
                    },
                    'training': {
                        'epochs': form_widgets['epochs'].value,
                        'batch_size': form_widgets['batch_size'].value,
                        'learning_rate': form_widgets['learning_rate'].value,
                        'optimizer': form_widgets['optimizer'].value,
                        'scheduler': form_widgets['scheduler'].value,
                        'weight_decay': form_widgets['weight_decay'].value,
                        'mixed_precision': form_widgets['mixed_precision'].value,
                        'save_period': form_widgets['save_period'].value,
                        'early_stopping': {
                            'enabled': form_widgets['early_stopping'].value,
                            'patience': form_widgets['early_stopping_patience'].value
                        }
                    },
                    'data': {
                        'val_split': form_widgets['val_split'].value,
                        'workers': form_widgets['workers'].value
                    },
                    'ui': {
                        'show_advanced_options': form_widgets['show_advanced'].value,
                        'enable_live_charts': form_widgets['enable_charts'].value
                    }
                }
            except Exception as e:
                logger.error(f"Failed to get form values: {e}")
                return {}
        
        # Add method to update form from config
        def update_from_config(config: Dict[str, Any]) -> None:
            """Update form values from configuration."""
            try:
                model_selection = config.get('model_selection', {})
                training_config = config.get('training', {})
                data_config = config.get('data', {})
                ui_config = config.get('ui', {})
                early_stopping = training_config.get('early_stopping', {})
                
                form_widgets['model_source'].value = model_selection.get('source', 'backbone')
                form_widgets['checkpoint_path'].value = model_selection.get('checkpoint_path', '')
                form_widgets['epochs'].value = training_config.get('epochs', 100)
                form_widgets['batch_size'].value = training_config.get('batch_size', 16)
                form_widgets['learning_rate'].value = training_config.get('learning_rate', 0.001)
                form_widgets['optimizer'].value = training_config.get('optimizer', 'adam')
                form_widgets['scheduler'].value = training_config.get('scheduler', 'cosine')
                form_widgets['weight_decay'].value = training_config.get('weight_decay', 0.0005)
                form_widgets['mixed_precision'].value = training_config.get('mixed_precision', True)
                form_widgets['early_stopping'].value = early_stopping.get('enabled', True)
                form_widgets['early_stopping_patience'].value = early_stopping.get('patience', 15)
                form_widgets['save_period'].value = training_config.get('save_period', 10)
                form_widgets['val_split'].value = data_config.get('val_split', 0.2)
                form_widgets['workers'].value = data_config.get('workers', 4)
                form_widgets['show_advanced'].value = ui_config.get('show_advanced_options', False)
                form_widgets['enable_charts'].value = ui_config.get('enable_live_charts', True)
                
                logger.debug("✅ Form updated from configuration")
                
            except Exception as e:
                logger.error(f"Failed to update form from config: {e}")
        
        # Attach methods to form
        form_accordion.get_form_values = get_form_values
        form_accordion.update_from_config = update_from_config
        form_accordion._widgets = form_widgets
        
        # Set up dynamic visibility for checkpoint path
        def on_model_source_change(change):
            """Show/hide checkpoint path based on model source."""
            if change['new'] == 'checkpoint':
                checkpoint_path_text.layout.display = 'block'
            else:
                checkpoint_path_text.layout.display = 'none'
        
        model_source_dropdown.observe(on_model_source_change, names='value')
        
        # Initial visibility setup
        if model_source_dropdown.value != 'checkpoint':
            checkpoint_path_text.layout.display = 'none'
        
        logger.debug("✅ Training configuration form created")
        return form_accordion
        
    except Exception as e:
        logger.error(f"Failed to create training form: {e}")
        raise


def create_simple_training_form(config: Dict[str, Any]) -> widgets.Widget:
    """Create a simplified training form for basic configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Simplified form widget
    """
    training_config = config.get('training', {})
    
    epochs = widgets.IntSlider(
        description="Epochs:",
        value=training_config.get('epochs', 100),
        min=1, max=500, step=10,
        style={'description_width': '120px'}
    )
    
    batch_size = widgets.IntSlider(
        description="Batch Size:",
        value=training_config.get('batch_size', 16),
        min=1, max=128, step=1,
        style={'description_width': '120px'}
    )
    
    lr = widgets.FloatLogSlider(
        description="Learning Rate:",
        value=training_config.get('learning_rate', 0.001),
        base=10, min=-5, max=-1, step=0.1,
        style={'description_width': '120px'}
    )
    
    return widgets.VBox([
        widgets.HTML("<h4>🚀 Quick Training Setup</h4>"),
        epochs, batch_size, lr
    ])