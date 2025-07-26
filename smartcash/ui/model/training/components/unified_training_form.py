"""
File: smartcash/ui/model/training/components/unified_training_form.py
Description: Simplified training form for unified training pipeline integration.
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.logger import get_module_logger


def create_unified_training_form(config: Dict[str, Any]) -> widgets.Widget:
    """Create simplified training form that matches unified_training_example.py configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Widget containing the unified training form
    """
    logger = get_module_logger("smartcash.ui.model.training.components")
    
    try:
        # Extract config sections with defaults matching unified_training_example.py
        training_config = config.get('training', {})
        
        # Backbone selection (limited to efficientnet_b4 and yolov5s as requested)
        backbone_dropdown = widgets.Dropdown(
            description="Backbone:",
            options=[
                ('EfficientNet-B4', 'efficientnet_b4'),
                ('YOLOv5s (CSPDarkNet)', 'cspdarknet')
            ],
            value=training_config.get('backbone', 'cspdarknet'),
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        # Training mode selection
        training_mode_dropdown = widgets.Dropdown(
            description="Training Mode:",
            options=[
                ('Two-Phase (Freeze → Fine-tune)', 'two_phase'),
                ('Single-Phase (Unified)', 'single_phase')
            ],
            value=training_config.get('training_mode', 'two_phase'),
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        # Phase epochs
        phase1_epochs = widgets.IntSlider(
            description="Phase 1 Epochs:",
            value=training_config.get('phase_1_epochs', 1),
            min=20, max=200, step=1,
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        phase2_epochs = widgets.IntSlider(
            description="Phase 2 Epochs:",
            value=training_config.get('phase_2_epochs', 1),
            min=50, max=200, step=1,
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        # Loss type
        loss_type_dropdown = widgets.Dropdown(
            description="Loss Type:",
            options=[
                ('Uncertainty Multi-Task', 'uncertainty_multi_task'),
                ('Weighted Multi-Task', 'weighted_multi_task'),
                ('Focal', 'focal'),
                ('Standard', 'standard')
            ],
            value=training_config.get('loss_type', 'uncertainty_multi_task'),
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        # Learning rates
        head_lr_p1 = widgets.FloatLogSlider(
            description="Head LR (P1):",
            value=training_config.get('head_lr_p1', 1e-3),
            base=10, min=-6, max=-1, step=0.1,
            readout_format='.1e',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        head_lr_p2 = widgets.FloatLogSlider(
            description="Head LR (P2):",
            value=training_config.get('head_lr_p2', 1e-4),
            base=10, min=-6, max=-1, step=0.1,
            readout_format='.1e',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        backbone_lr = widgets.FloatLogSlider(
            description="Backbone LR:",
            value=training_config.get('backbone_lr', 1e-5),
            base=10, min=-7, max=-3, step=0.1,
            readout_format='.1e',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        # Batch size with auto-detection option
        batch_size_text = widgets.Text(
            description="Batch Size:",
            value=str(training_config.get('batch_size', 'auto')),
            placeholder="auto (platform optimized) or number",
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        # Early stopping configuration
        early_stopping_checkbox = widgets.Checkbox(
            description="Enable Early Stopping",
            value=training_config.get('early_stopping_enabled', True),
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        patience = widgets.IntSlider(
            description="Patience:",
            value=training_config.get('early_stopping_patience', 15),
            min=3, max=50, step=1,
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        es_metric = widgets.Dropdown(
            description="ES Metric:",
            options=[
                ('Validation mAP@0.5', 'val_map50'),
                ('Validation Loss', 'val_loss'),
                ('Training Loss', 'train_loss'),
                ('Validation mAP@0.75', 'val_map75'),
                ('Validation Accuracy', 'val_accuracy')
            ],
            value=training_config.get('early_stopping_metric', 'val_map50'),
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        es_mode = widgets.Dropdown(
            description="ES Mode:",
            options=[
                ('Maximize (for mAP, accuracy)', 'max'),
                ('Minimize (for loss)', 'min')
            ],
            value=training_config.get('early_stopping_mode', 'max'),
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        min_delta = widgets.FloatText(
            description="Min Delta:",
            value=training_config.get('early_stopping_min_delta', 0.001),
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        # Single-phase specific options
        single_layer_mode = widgets.Dropdown(
            description="Layer Mode:",
            options=[
                ('Multi-layer (all layers)', 'multi'),
                ('Single-layer (layer_1 only)', 'single')
            ],
            value=training_config.get('single_phase_layer_mode', 'multi'),
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        single_freeze_backbone = widgets.Checkbox(
            description="Freeze Backbone (Single-phase)",
            value=training_config.get('single_phase_freeze_backbone', False),
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        # System options
        force_cpu = widgets.Checkbox(
            description="Force CPU Training",
            value=training_config.get('force_cpu', False),
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        verbose = widgets.Checkbox(
            description="Verbose Logging",
            value=training_config.get('verbose', True),
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        # Checkpoint directory
        checkpoint_dir = widgets.Text(
            description="Checkpoint Dir:",
            value=training_config.get('checkpoint_dir', 'data/checkpoints'),
            style={'description_width': '120px'},
            layout=widgets.Layout(width='auto')
        )
        
        # Create form sections in two columns
        # Basic section - left column
        basic_left_column = widgets.VBox([
            widgets.HTML("<h4 style='color: #007bff; margin: 10px 0;'>🏗️ Model & Training Mode</h4>"),
            backbone_dropdown,
            training_mode_dropdown,
            widgets.HTML("<h4 style='color: #28a745; margin: 15px 0 10px 0;'>📊 Training Phases</h4>"),
            phase1_epochs,
            phase2_epochs
        ], layout=widgets.Layout(padding='10px', width='48%'))
        
        # Basic section - right column
        basic_right_column = widgets.VBox([
            widgets.HTML("<h4 style='color: #fd7e14; margin: 10px 0;'>🎯 Learning Configuration</h4>"),
            loss_type_dropdown,
            head_lr_p1,
            head_lr_p2,
            backbone_lr,
            batch_size_text
        ], layout=widgets.Layout(padding='10px', width='48%'))
        
        # Basic section - combine columns
        basic_section = widgets.HBox([
            basic_left_column,
            basic_right_column
        ], layout=widgets.Layout(width='100%', justify_content='space-between'))
        
        # Advanced section - left column
        advanced_left_column = widgets.VBox([
            widgets.HTML("<h4 style='color: #dc3545; margin: 10px 0;'>⏹️ Early Stopping</h4>"),
            early_stopping_checkbox,
            patience,
            es_metric,
            es_mode,
            min_delta
        ], layout=widgets.Layout(padding='10px', width='48%'))
        
        # Single-phase options header
        single_phase_options = widgets.HTML("<h4 style='color: #6f42c1; margin: 10px 0;'>🔧 Single-Phase Options</h4>")
        
        # Advanced section - right column
        advanced_right_column = widgets.VBox([
            single_phase_options,
            single_layer_mode,
            single_freeze_backbone,
            widgets.HTML("<h4 style='color: #17a2b8; margin: 15px 0 10px 0;'>⚙️ System Options</h4>"),
            force_cpu,
            verbose,
            checkpoint_dir
        ], layout=widgets.Layout(padding='10px', width='48%'))
        
        # Advanced section - combine columns
        advanced_section = widgets.HBox([
            advanced_left_column,
            advanced_right_column
        ], layout=widgets.Layout(width='100%', justify_content='space-between'))
        
        # Create accordion
        form_accordion = widgets.Accordion(children=[basic_section, advanced_section])
        form_accordion.set_title(0, "🚀 Basic Training Configuration")
        form_accordion.set_title(1, "⚙️ Advanced Options")
        form_accordion.selected_index = 0  # Open basic section by default
        
        # Store widget references
        form_widgets = {
            'backbone': backbone_dropdown,
            'training_mode': training_mode_dropdown,
            'phase_1_epochs': phase1_epochs,
            'phase_2_epochs': phase2_epochs,
            'loss_type': loss_type_dropdown,
            'head_lr_p1': head_lr_p1,
            'head_lr_p2': head_lr_p2,
            'backbone_lr': backbone_lr,
            'batch_size': batch_size_text,
            'early_stopping_enabled': early_stopping_checkbox,
            'early_stopping_patience': patience,
            'early_stopping_metric': es_metric,
            'early_stopping_mode': es_mode,
            'early_stopping_min_delta': min_delta,
            'single_phase_layer_mode': single_layer_mode,
            'single_phase_freeze_backbone': single_freeze_backbone,
            'single_phase_options': single_phase_options,
            'force_cpu': force_cpu,
            'verbose': verbose,
            'checkpoint_dir': checkpoint_dir
        }
        
        def get_form_values() -> Dict[str, Any]:
            """Extract current form values matching unified training pipeline parameters."""
            try:
                # Parse batch size
                batch_size_value = form_widgets['batch_size'].value.strip()
                if batch_size_value.lower() == 'auto' or not batch_size_value:
                    batch_size = None  # Auto-detection
                else:
                    batch_size = int(batch_size_value)
                
                return {
                    'backbone': form_widgets['backbone'].value,
                    'training_mode': form_widgets['training_mode'].value,
                    'phase_1_epochs': form_widgets['phase_1_epochs'].value,
                    'phase_2_epochs': form_widgets['phase_2_epochs'].value,
                    'loss_type': form_widgets['loss_type'].value,
                    'head_lr_p1': form_widgets['head_lr_p1'].value,
                    'head_lr_p2': form_widgets['head_lr_p2'].value,
                    'backbone_lr': form_widgets['backbone_lr'].value,
                    'batch_size': batch_size,
                    'early_stopping_enabled': form_widgets['early_stopping_enabled'].value,
                    'early_stopping_patience': form_widgets['early_stopping_patience'].value,
                    'early_stopping_metric': form_widgets['early_stopping_metric'].value,
                    'early_stopping_mode': form_widgets['early_stopping_mode'].value,
                    'early_stopping_min_delta': form_widgets['early_stopping_min_delta'].value,
                    'single_phase_layer_mode': form_widgets['single_phase_layer_mode'].value,
                    'single_phase_freeze_backbone': form_widgets['single_phase_freeze_backbone'].value,
                    'force_cpu': form_widgets['force_cpu'].value,
                    'verbose': form_widgets['verbose'].value,
                    'checkpoint_dir': form_widgets['checkpoint_dir'].value
                }
            except Exception as e:
                logger.error(f"Failed to get form values: {e}")
                return {}
        
        def update_from_config(config: Dict[str, Any]) -> None:
            """Update form values from configuration."""
            try:
                training_config = config.get('training', {})
                
                form_widgets['backbone'].value = training_config.get('backbone', 'cspdarknet')
                form_widgets['training_mode'].value = training_config.get('training_mode', 'two_phase')
                form_widgets['phase_1_epochs'].value = training_config.get('phase_1_epochs', 1)
                form_widgets['phase_2_epochs'].value = training_config.get('phase_2_epochs', 1)
                form_widgets['loss_type'].value = training_config.get('loss_type', 'uncertainty_multi_task')
                form_widgets['head_lr_p1'].value = training_config.get('head_lr_p1', 1e-3)
                form_widgets['head_lr_p2'].value = training_config.get('head_lr_p2', 1e-4)
                form_widgets['backbone_lr'].value = training_config.get('backbone_lr', 1e-5)
                
                # Handle batch size
                batch_size = training_config.get('batch_size')
                if batch_size is None:
                    form_widgets['batch_size'].value = 'auto'
                else:
                    form_widgets['batch_size'].value = str(batch_size)
                
                form_widgets['early_stopping_enabled'].value = training_config.get('early_stopping_enabled', True)
                form_widgets['early_stopping_patience'].value = training_config.get('early_stopping_patience', 15)
                form_widgets['early_stopping_metric'].value = training_config.get('early_stopping_metric', 'val_map50')
                form_widgets['early_stopping_mode'].value = training_config.get('early_stopping_mode', 'max')
                form_widgets['early_stopping_min_delta'].value = training_config.get('early_stopping_min_delta', 0.001)
                form_widgets['single_phase_layer_mode'].value = training_config.get('single_phase_layer_mode', 'multi')
                form_widgets['single_phase_freeze_backbone'].value = training_config.get('single_phase_freeze_backbone', False)
                form_widgets['force_cpu'].value = training_config.get('force_cpu', False)
                form_widgets['verbose'].value = training_config.get('verbose', True)
                form_widgets['checkpoint_dir'].value = training_config.get('checkpoint_dir', 'data/checkpoints')
                
                logger.debug("✅ Unified form updated from configuration")
                
            except Exception as e:
                logger.error(f"Failed to update unified form from config: {e}")
        
        # Dynamic visibility for single-phase options
        def on_training_mode_change(change):
            """Update single-phase options visibility based on training mode."""
            is_single_phase = change['new'] == 'single_phase'
            
            # Show/hide single-phase specific options
            single_layer_mode.layout.display = 'block' if is_single_phase else 'none'
            single_freeze_backbone.layout.display = 'block' if is_single_phase else 'none'
        
        training_mode_dropdown.observe(on_training_mode_change, names='value')
        
        # Dynamic visibility for early stopping options
        def on_early_stopping_change(change):
            """Update early stopping options visibility."""
            is_enabled = change['new']
            
            patience.layout.display = 'block' if is_enabled else 'none'
            es_metric.layout.display = 'block' if is_enabled else 'none'
            es_mode.layout.display = 'block' if is_enabled else 'none'
            min_delta.layout.display = 'block' if is_enabled else 'none'
        
        early_stopping_checkbox.observe(on_early_stopping_change, names='value')
        
        # Initial visibility setup
        is_single_phase = training_mode_dropdown.value == 'single_phase'
        single_phase_options.layout.display = 'block' if is_single_phase else 'none'
        single_layer_mode.layout.display = 'block' if is_single_phase else 'none'
        single_freeze_backbone.layout.display = 'block' if is_single_phase else 'none'
        
        is_early_stopping = early_stopping_checkbox.value
        patience.layout.display = 'block' if is_early_stopping else 'none'
        es_metric.layout.display = 'block' if is_early_stopping else 'none'
        es_mode.layout.display = 'block' if is_early_stopping else 'none'
        min_delta.layout.display = 'block' if is_early_stopping else 'none'
        
        # Attach methods to form
        form_accordion.get_form_values = get_form_values
        form_accordion.update_from_config = update_from_config
        form_accordion._widgets = form_widgets
        
        logger.debug("✅ Unified training configuration form created")
        return form_accordion
        
    except Exception as e:
        logger.error(f"Failed to create unified training form: {e}")
        raise