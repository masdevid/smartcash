"""
File: smartcash/ui/model/train/components/training_ui.py
Description: Model training UI following SmartCash standardized template.

This module provides the user interface for training YOLOv5 models
with real-time monitoring, metrics visualization, and training control.

Container Order:
1. Header Container (Title, Status)
2. Form Container (Training Configuration)
3. Action Container (Start/Stop/Resume/Validate Buttons)
4. Summary Container (Training Overview)
5. Operation Container (Progress + Logs)
6. Footer Container (Tips and Info)
"""

from typing import Optional, Dict, Any
import ipywidgets as widgets

# Core container imports - standardized across all modules
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container, LayoutType
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.core.errors.handlers import handle_ui_errors

# Module imports
from ..constants import UI_CONFIG, BUTTON_CONFIG, DEFAULT_CONFIG, VALIDATION_RULES

# Module constants (for validator compliance)
UI_CONFIG = UI_CONFIG
BUTTON_CONFIG = BUTTON_CONFIG


@handle_ui_errors(error_component_title="Training UI Error")
def create_training_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create the model training UI following SmartCash standards.
    
    This function creates a complete UI for training YOLOv5 models
    with the following sections:
    - Training configuration parameters
    - Training control operations
    - Real-time progress monitoring
    - Metrics visualization and tracking
    
    Args:
        config: Optional configuration dictionary for the UI
        **kwargs: Additional keyword arguments passed to UI components
        
    Returns:
        Dictionary containing all UI components and their references with 'ui_components' key
        
    Example:
        >>> ui = create_training_ui()
        >>> display(ui['ui'])  # Display the UI
    """
    # Initialize configuration and components dictionary
    current_config = config or DEFAULT_CONFIG.copy()
    ui_components = {
        'config': current_config,
        'containers': {},
        'widgets': {}
    }
    
    # === 1. Create Header Container ===
    header_container = create_header_container(
        title=f"{UI_CONFIG['icon']} {UI_CONFIG['title']}",
        subtitle=UI_CONFIG['subtitle'],
        status_message="Ready to start training",
        status_type="info"
    )
    # Store both the container object and its widget
    ui_components['containers']['header'] = {
        'container': header_container.container,
        'widget': header_container
    }
    
    # === 2. Create Form Container ===
    # Create form widgets
    form_widgets = _create_module_form_widgets(current_config)
    
    # Create form container with the widgets
    form_container = create_form_container(
        form_rows=form_widgets['form_rows'],
        layout_type=LayoutType.COLUMN,
        container_margin="0",
        container_padding="16px",
        gap="12px"
    )
    
    # Store references
    ui_components['containers']['form'] = form_container
    ui_components['widgets'].update(form_widgets['widgets'])
    
    # === 3. Create Action Container ===
    # Create action buttons from BUTTON_CONFIG
    action_buttons = []
    for button_id, btn_config in BUTTON_CONFIG.items():
        action_buttons.append({
            'name': button_id,
            'label': btn_config['text'],
            'button_style': btn_config['style'],
            'tooltip': btn_config['tooltip'],
            'icon': 'play' if button_id == 'start' else 'stop' if button_id == 'stop' else 'refresh' if button_id == 'resume' else 'check'
        })
    
    action_container = create_action_container(
        buttons=action_buttons,
        title="🎯 Training Operations",
        alignment="left"
    )
    
    # Store references
    ui_components['containers']['actions'] = action_container
    
    # Store individual button references for easier access
    if hasattr(action_container, 'get'):
        for btn in action_buttons:
            button_ref = action_container.get(btn['name'])
            if button_ref:
                ui_components['widgets'][f'{btn["name"]}_button'] = button_ref
                # Also store directly under containers.actions for handler access
                ui_components['containers']['actions'][btn['name']] = button_ref
    
    # === 4. Create Summary Container ===
    summary_content = _create_module_summary_content(current_config)
    summary_container = create_summary_container(
        title="📊 Training Overview",
        theme="primary",
        icon="🚀"
    )
    summary_container.set_content(summary_content)
    
    ui_components['containers']['summary'] = summary_container
    
    # === 5. Create Operation Container ===
    operation_container = create_operation_container(
        title="🔄 Training Progress",
        show_progress=True,
        show_logs=True,
        collapsible=True,
        collapsed=False
    )
    ui_components['containers']['operation'] = operation_container
    
    # === 6. Create Footer Container ===
    footer_container = create_footer_container(
        info_box=_create_module_info_box(),
        show_tips=True,
        show_version=True
    )
    # Store both the container object and its widget
    ui_components['containers']['footer'] = {
        'container': footer_container.container,
        'widget': footer_container
    }
    
    # === 7. Create Main Container ===
    main_container = create_main_container(
        header=ui_components['containers']['header']['container'],
        body=widgets.VBox([
            ui_components['containers']['form']['container'],
            ui_components['containers']['actions']['container'],
            ui_components['containers']['summary'].container,
            ui_components['containers']['operation']['container']
        ]),
        footer=ui_components['containers']['footer']['container'],
        container_config={
            'margin': '0 auto',
            'max_width': '1200px',
            'padding': '10px',
            'border': '1px solid #e0e0e0',
            'border_radius': '5px',
            'box_shadow': '0 1px 3px rgba(0,0,0,0.1)'
        }
    )
    
    # Store main UI references
    ui_components['ui'] = main_container.container
    ui_components['main_container'] = main_container
    
    result = {
        'ui_components': ui_components,
        'ui': ui_components['ui']
    }
    
    # Add all components to the root for backward compatibility
    result.update(ui_components['containers'])
    result.update(ui_components['widgets'])
    
    # Add legacy compatibility
    result.update({
        'ui_initialized': True,
        'module_name': 'train',
        'parent_module': 'model'
    })
    
    return result


def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create module-specific form widgets for training configuration.
    
    Args:
        config: Configuration dictionary for the form widgets
        
    Returns:
        Dictionary containing the form UI and widget references
    """
    training_config = config.get("training", {})
    optimizer_config = config.get("optimizer", {})
    scheduler_config = config.get("scheduler", {})
    
    # Training Parameters Section
    epochs_input = widgets.IntSlider(
        value=training_config.get('epochs', 100),
        min=VALIDATION_RULES['epochs']['min'],
        max=VALIDATION_RULES['epochs']['max'],
        step=1,
        description='Epochs:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    batch_size_input = widgets.IntSlider(
        value=training_config.get('batch_size', 16),
        min=VALIDATION_RULES['batch_size']['min'],
        max=VALIDATION_RULES['batch_size']['max'],
        step=1,
        description='Batch Size:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    learning_rate_input = widgets.FloatLogSlider(
        value=training_config.get('learning_rate', 0.001),
        base=10,
        min=-6,
        max=0,
        step=0.1,
        description='Learning Rate:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    validation_interval_input = widgets.IntSlider(
        value=training_config.get('validation_interval', 1),
        min=VALIDATION_RULES['validation_interval']['min'],
        max=VALIDATION_RULES['validation_interval']['max'],
        step=1,
        description='Val Interval:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Optimizer Parameters Section
    optimizer_dropdown = widgets.Dropdown(
        options=[('Adam', 'adam'), ('SGD', 'sgd'), ('AdamW', 'adamw')],
        value=optimizer_config.get('type', 'adam'),
        description='Optimizer:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    weight_decay_input = widgets.FloatLogSlider(
        value=optimizer_config.get('weight_decay', 0.0005),
        base=10,
        min=-6,
        max=-1,
        step=0.1,
        description='Weight Decay:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Scheduler Parameters Section
    scheduler_dropdown = widgets.Dropdown(
        options=[('Cosine', 'cosine'), ('Step', 'step'), ('Exponential', 'exponential')],
        value=scheduler_config.get('type', 'cosine'),
        description='Scheduler:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    warmup_epochs_input = widgets.IntSlider(
        value=scheduler_config.get('warmup_epochs', 5),
        min=0,
        max=20,
        step=1,
        description='Warmup Epochs:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Advanced Options
    early_stopping_config = training_config.get('early_stopping', {})
    early_stopping_checkbox = widgets.Checkbox(
        value=early_stopping_config.get('enabled', True),
        description='Enable Early Stopping',
        layout=widgets.Layout(width='50%', margin='5px 0')
    )
    
    mixed_precision_config = config.get('mixed_precision', {})
    mixed_precision_checkbox = widgets.Checkbox(
        value=mixed_precision_config.get('enabled', True),
        description='Enable Mixed Precision',
        layout=widgets.Layout(width='50%', margin='5px 0')
    )
    
    # Resume Training
    checkpoint_path_input = widgets.Text(
        value='',
        description='Checkpoint Path:',
        placeholder='/data/checkpoints/best_model.pt',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Create form rows
    form_rows = [
        [widgets.HTML("<h4>⚙️ Training Parameters</h4>")],
        [widgets.HBox([epochs_input, batch_size_input])],
        [widgets.HBox([learning_rate_input, validation_interval_input])],
        [widgets.HTML("<h4>🔧 Optimizer Configuration</h4>")],
        [widgets.HBox([optimizer_dropdown, weight_decay_input])],
        [widgets.HTML("<h4>📈 Learning Rate Scheduler</h4>")],
        [widgets.HBox([scheduler_dropdown, warmup_epochs_input])],
        [widgets.HTML("<h4>🎛️ Advanced Options</h4>")],
        [widgets.HBox([early_stopping_checkbox, mixed_precision_checkbox])],
        [widgets.HTML("<h4>💾 Resume Training</h4>")],
        [checkpoint_path_input],
        [widgets.HTML("""
            <div style='margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px; font-size: 0.9em; color: #666;'>
                <strong>💡 Tips:</strong><br>
                • Start with default settings for initial training<br>
                • Reduce learning rate if loss oscillates<br>
                • Increase batch size if GPU memory allows<br>
                • Enable mixed precision to reduce memory usage
            </div>
        """)]
    ]
    
    return {
        'form_rows': form_rows,
        'widgets': {
            'epochs_input': epochs_input,
            'batch_size_input': batch_size_input,
            'learning_rate_input': learning_rate_input,
            'validation_interval_input': validation_interval_input,
            'optimizer_dropdown': optimizer_dropdown,
            'weight_decay_input': weight_decay_input,
            'scheduler_dropdown': scheduler_dropdown,
            'warmup_epochs_input': warmup_epochs_input,
            'early_stopping_checkbox': early_stopping_checkbox,
            'mixed_precision_checkbox': mixed_precision_checkbox,
            'checkpoint_path_input': checkpoint_path_input
        }
    }


def _create_module_summary_content(config: Dict[str, Any]) -> str:
    """
    Create summary content for the module.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HTML string containing the summary content
    """
    training_config = config.get("training", {})
    optimizer_config = config.get("optimizer", {})
    scheduler_config = config.get("scheduler", {})
    
    return f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    ">
        <h4 style="margin: 0 0 15px 0; font-size: 1.2rem; font-weight: 600;">
            🚀 Training Configuration
        </h4>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px;">
                <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Training</div>
                <div style="font-size: 1.1rem; font-weight: 600;">
                    {training_config.get('epochs', 100)} epochs
                </div>
                <div style="font-size: 0.8rem; opacity: 0.7;">
                    Batch: {training_config.get('batch_size', 16)} | 
                    LR: {training_config.get('learning_rate', 0.001)}
                </div>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px;">
                <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Optimizer</div>
                <div style="font-size: 1.1rem; font-weight: 600;">
                    {optimizer_config.get('type', 'adam').upper()}
                </div>
                <div style="font-size: 0.8rem; opacity: 0.7;">
                    Weight Decay: {optimizer_config.get('weight_decay', 0.0005)}
                </div>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px;">
                <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Scheduler</div>
                <div style="font-size: 1.1rem; font-weight: 600;">
                    {scheduler_config.get('type', 'cosine').title()}
                </div>
                <div style="font-size: 0.8rem; opacity: 0.7;">
                    Warmup: {scheduler_config.get('warmup_epochs', 5)} epochs
                </div>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px;">
                <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Features</div>
                <div style="font-size: 1.1rem; font-weight: 600;">
                    {'✅' if training_config.get('early_stopping', {}).get('enabled', True) else '❌'} Early Stop
                </div>
                <div style="font-size: 0.8rem; opacity: 0.7;">
                    {'✅' if config.get('mixed_precision', {}).get('enabled', True) else '❌'} Mixed Precision
                </div>
            </div>
        </div>
    </div>
    """


def _create_module_info_box() -> widgets.Widget:
    """
    Create the info box content for the footer.
    
    Returns:
        Widget containing the info box content
    """
    return widgets.HTML(
        value="""
        <div style="padding: 12px; background: #e3f2fd; border-radius: 4px; margin: 8px 0;">
            <h4 style="margin-top: 0; color: #0d47a1;">🚀 Training Guide</h4>
            <p>This module helps you train YOLOv5 models with real-time monitoring and control.</p>
            <ol style="margin: 8px 0 0 16px; padding-left: 8px;">
                <li>Configure training parameters (epochs, batch size, learning rate)</li>
                <li>Set up optimizer and scheduler options</li>
                <li>Enable advanced features (early stopping, mixed precision)</li>
                <li>Click 'Start Training' to begin model training</li>
                <li>Monitor progress and metrics in real-time</li>
                <li>Use 'Stop' or 'Resume' to control training process</li>
            </ol>
            <div style="margin-top: 8px; padding: 6px; background: rgba(0,0,0,0.05); border-radius: 3px;">
                <strong>💡 Tip:</strong> Monitor loss curves for signs of overfitting and adjust parameters accordingly
            </div>
        </div>
        """
    )