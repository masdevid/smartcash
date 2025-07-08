"""
File: smartcash/ui/model/train/components/training_ui.py
UI components for training module following container standards.
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets

from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.chart_container import create_chart_container

from ..constants import UI_CONFIG, DEFAULT_CONFIG, VALIDATION_RULES


def create_training_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create training UI using standard container components.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing UI components
    """
    config = config or DEFAULT_CONFIG.copy()
    ui_components = {}
    
    # 1. Header Container
    header_container = create_header_container(
        title=UI_CONFIG["title"],
        subtitle=UI_CONFIG["subtitle"], 
        icon=UI_CONFIG["icon"]
    )
    ui_components['header_container'] = header_container.container
    
    # 2. Training Config Summary (Custom Modern Style)
    config_summary = create_training_config_summary(config)
    ui_components['config_summary'] = config_summary
    
    # 3. Form Container with training configuration
    form_container = create_form_container()
    input_components = create_training_config_form(config)
    form_container['add_item'](input_components['ui'])
    ui_components['form_container'] = form_container['container']
    ui_components['input_options'] = input_components
    
    # 4. Action Container with training operations
    action_container = create_action_container(
        buttons=[
            {
                "button_id": "start",
                "text": "🚀 Start Training",
                "style": "success",
                "order": 1,
                "tooltip": "Start model training with current configuration"
            },
            {
                "button_id": "stop", 
                "text": "🛑 Stop Training",
                "style": "danger",
                "order": 2,
                "tooltip": "Stop current training process"
            },
            {
                "button_id": "resume",
                "text": "🔄 Resume Training", 
                "style": "warning",
                "order": 3,
                "tooltip": "Resume training from checkpoint"
            }
        ],
        title="🎯 Training Operations",
        alignment="center"
    )
    ui_components['action_container'] = action_container['container']
    ui_components['start_button'] = action_container['buttons'].get('start')
    ui_components['stop_button'] = action_container['buttons'].get('stop')
    ui_components['resume_button'] = action_container['buttons'].get('resume')
    
    # 5. Operation Container with progress and logging
    operation_container = create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        log_module_name="training"
    )
    ui_components['operation_container'] = operation_container['container']
    ui_components['progress_tracker'] = operation_container['progress_tracker']
    ui_components['log_output'] = operation_container['log_accordion']
    ui_components['log_accordion'] = operation_container['log_accordion']
    
    # 6. Chart Container with dual charts for metrics
    chart_container = create_chart_container(
        title="📊 Training Metrics",
        chart_type="line",
        columns=2,
        height=350
    )
    ui_components['chart_container'] = chart_container
    
    # 7. Footer Container with training info
    info_box = create_training_info_box()
    footer_container = widgets.VBox([info_box], layout=widgets.Layout(
        width='100%',
        margin='15px 0 0 0'
    ))
    ui_components['footer_container'] = footer_container
    
    # 8. Main Container Assembly
    main_container = widgets.VBox([
        ui_components['header_container'],
        ui_components['config_summary'],
        ui_components['form_container'],
        ui_components['action_container'],
        ui_components['operation_container'],
        chart_container.container,
        ui_components['footer_container']
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    ui_components['ui'] = main_container
    ui_components['main_container'] = main_container
    
    # Metadata
    ui_components.update({
        'ui_initialized': True,
        'module_name': 'train',
        'parent_module': 'model'
    })
    
    return ui_components


def create_training_config_summary(config: Dict[str, Any]) -> widgets.Widget:
    """
    Create custom modern style training configuration summary.
    
    Args:
        config: Training configuration
        
    Returns:
        Widget containing configuration summary
    """
    training_config = config.get("training", {})
    optimizer_config = config.get("optimizer", {})
    scheduler_config = config.get("scheduler", {})
    
    summary_html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
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
    
    return widgets.HTML(summary_html)


def create_training_config_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create training configuration form.
    
    Args:
        config: Training configuration
        
    Returns:
        Dictionary containing form components
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
        layout={'width': '100%'}
    )
    
    batch_size_input = widgets.IntSlider(
        value=training_config.get('batch_size', 16),
        min=VALIDATION_RULES['batch_size']['min'],
        max=VALIDATION_RULES['batch_size']['max'],
        step=1,
        description='Batch Size:',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    learning_rate_input = widgets.FloatLogSlider(
        value=training_config.get('learning_rate', 0.001),
        base=10,
        min=-6,
        max=0,
        step=0.1,
        description='Learning Rate:',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    validation_interval_input = widgets.IntSlider(
        value=training_config.get('validation_interval', 1),
        min=VALIDATION_RULES['validation_interval']['min'],
        max=VALIDATION_RULES['validation_interval']['max'],
        step=1,
        description='Val Interval:',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    # Optimizer Parameters Section
    optimizer_dropdown = widgets.Dropdown(
        options=[('Adam', 'adam'), ('SGD', 'sgd'), ('AdamW', 'adamw')],
        value=optimizer_config.get('type', 'adam'),
        description='Optimizer:',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    weight_decay_input = widgets.FloatLogSlider(
        value=optimizer_config.get('weight_decay', 0.0005),
        base=10,
        min=-6,
        max=-1,
        step=0.1,
        description='Weight Decay:',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    # Scheduler Parameters Section
    scheduler_dropdown = widgets.Dropdown(
        options=[('Cosine', 'cosine'), ('Step', 'step'), ('Exponential', 'exponential')],
        value=scheduler_config.get('type', 'cosine'),
        description='Scheduler:',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    warmup_epochs_input = widgets.IntSlider(
        value=scheduler_config.get('warmup_epochs', 5),
        min=0,
        max=20,
        step=1,
        description='Warmup Epochs:',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    # Early Stopping Section
    early_stopping_config = training_config.get('early_stopping', {})
    early_stopping_enabled = widgets.Checkbox(
        value=early_stopping_config.get('enabled', True),
        description='Enable Early Stopping',
        style={'description_width': 'initial'},
        layout={'width': '100%'}
    )
    
    early_stopping_patience = widgets.IntSlider(
        value=early_stopping_config.get('patience', 15),
        min=5,
        max=50,
        step=1,
        description='Patience:',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    # Mixed Precision Section
    mixed_precision_config = config.get('mixed_precision', {})
    mixed_precision_enabled = widgets.Checkbox(
        value=mixed_precision_config.get('enabled', True),
        description='Enable Mixed Precision',
        style={'description_width': 'initial'},
        layout={'width': '100%'}
    )
    
    # Checkpoint Section
    checkpoint_path_input = widgets.Text(
        value='',
        description='Checkpoint Path:',
        placeholder='/data/checkpoints/best_model.pt',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    # Create form layout
    form_ui = widgets.VBox([
        widgets.HTML("<h4>⚙️ Training Parameters</h4>"),
        widgets.HBox([
            widgets.VBox([epochs_input, learning_rate_input], layout={'width': '50%'}),
            widgets.VBox([batch_size_input, validation_interval_input], layout={'width': '50%'})
        ], layout={'width': '100%'}),
        
        widgets.HTML("<h4 style='margin-top: 20px;'>🔧 Optimizer Configuration</h4>"),
        widgets.HBox([
            widgets.VBox([optimizer_dropdown], layout={'width': '50%'}),
            widgets.VBox([weight_decay_input], layout={'width': '50%'})
        ], layout={'width': '100%'}),
        
        widgets.HTML("<h4 style='margin-top: 20px;'>📈 Learning Rate Scheduler</h4>"),
        widgets.HBox([
            widgets.VBox([scheduler_dropdown], layout={'width': '50%'}),
            widgets.VBox([warmup_epochs_input], layout={'width': '50%'})
        ], layout={'width': '100%'}),
        
        widgets.HTML("<h4 style='margin-top: 20px;'>🎛️ Advanced Options</h4>"),
        widgets.HBox([
            widgets.VBox([early_stopping_enabled, early_stopping_patience], layout={'width': '50%'}),
            widgets.VBox([mixed_precision_enabled], layout={'width': '50%'})
        ], layout={'width': '100%'}),
        
        widgets.HTML("<h4 style='margin-top: 20px;'>💾 Resume Training</h4>"),
        checkpoint_path_input,
        
        widgets.HTML(
            "<div style='margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px; font-size: 0.9em; color: #666;'>"
            "<strong>💡 Tips:</strong><br>"
            "• Start with default settings for initial training<br>"
            "• Reduce learning rate if loss oscillates<br>"
            "• Increase batch size if GPU memory allows<br>"
            "• Enable mixed precision to reduce memory usage"
            "</div>"
        )
    ])
    
    return {
        'ui': form_ui,
        'epochs_input': epochs_input,
        'batch_size_input': batch_size_input,
        'learning_rate_input': learning_rate_input,
        'validation_interval_input': validation_interval_input,
        'optimizer_dropdown': optimizer_dropdown,
        'weight_decay_input': weight_decay_input,
        'scheduler_dropdown': scheduler_dropdown,
        'warmup_epochs_input': warmup_epochs_input,
        'early_stopping_enabled': early_stopping_enabled,
        'early_stopping_patience': early_stopping_patience,
        'mixed_precision_enabled': mixed_precision_enabled,
        'checkpoint_path_input': checkpoint_path_input
    }


def create_training_info_box() -> widgets.Widget:
    """
    Create info box for footer container.
    
    Returns:
        Widget containing helpful training information
    """
    info_html = """
    <div class="alert alert-info" style="font-size: 0.9em; padding: 12px; margin: 0;">
        <strong>🎯 Training Pipeline Information:</strong>
        <ul style="margin: 8px 0 0 15px; padding: 0;">
            <li><strong>Loss Chart:</strong> Monitors training and validation loss over time</li>
            <li><strong>Performance Chart:</strong> Tracks mAP, accuracy, precision, and F1 scores</li>
            <li><strong>Early Stopping:</strong> Automatically stops when validation metrics plateau</li>
            <li><strong>Mixed Precision:</strong> Reduces memory usage while maintaining accuracy</li>
            <li><strong>Checkpoints:</strong> Best models are automatically saved during training</li>
        </ul>
        <div style="margin-top: 10px; padding: 8px; background: rgba(0,0,0,0.05); border-radius: 3px;">
            <strong>⚡ Performance Tips:</strong> Use GPU acceleration, monitor metrics regularly, and save checkpoints frequently
        </div>
    </div>
    """
    
    return widgets.HTML(info_html)