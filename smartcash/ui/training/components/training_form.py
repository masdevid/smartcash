"""
File: smartcash/ui/training/components/training_form.py
Deskripsi: Enhanced training form dengan YAML config integration dan consolidated components
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_training_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create training form dengan YAML config integration dan existing components"""
    try:
        from smartcash.ui.components.progress_tracking import create_progress_tracking_container
        from smartcash.ui.components.log_accordion import create_log_accordion
        from smartcash.ui.components.status_panel import create_status_panel
        
        # Control buttons dengan YAML-based model info
        control_buttons = _create_enhanced_control_buttons(config)
        
        # Progress tracking dengan existing component
        progress_components = create_progress_tracking_container()
        
        # Status panel dengan model-specific message
        model_type = config.get('model', {}).get('type', 'efficient_basic')
        backbone = config.get('model', {}).get('backbone', 'efficientnet_b4')
        status_message = f"🧠 {backbone.upper()} ({model_type}) siap untuk training"
        status_panel = create_status_panel(status_message, "info")
        
        # Log accordion dengan training namespace
        log_components = create_log_accordion('training', height='250px')
        
        # Config tabs dengan YAML integration
        config_tabs = _create_yaml_config_tabs(config)
        
        # Chart & metrics outputs
        chart_output = widgets.Output(layout=widgets.Layout(max_height='400px', overflow='auto'))
        metrics_output = widgets.Output(layout=widgets.Layout(max_height='150px', overflow='auto'))
        
        return {
            **control_buttons,
            'progress_container': progress_components.get('container'),
            'progress_tracker': progress_components.get('tracker'),
            'status_panel': status_panel,
            'config_tabs': config_tabs,
            'info_display': config_tabs,
            'log_output': log_components.get('log_output'),
            'log_accordion': log_components.get('log_accordion'),
            'chart_output': chart_output,
            'metrics_output': metrics_output,
            'config': config
        }
        
    except Exception as e:
        return _create_simple_fallback(str(e))

def _create_enhanced_control_buttons(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create enhanced control buttons dengan YAML config awareness"""
    from smartcash.ui.components.action_buttons import create_action_buttons
    
    # Ambil info dari YAML config untuk button labels
    model_type = config.get('model', {}).get('type', 'efficient_basic')
    epochs = config.get('epochs', 100)
    
    # Primary button dengan model-specific label
    primary_buttons = create_action_buttons(
        primary_label=f"🚀 Train {model_type.title()} ({epochs} epochs)",
        primary_icon="",
        primary_style='success',
        secondary_buttons=[],
        cleanup_enabled=False,
        button_width='200px'
    )
    
    # Control buttons
    stop_button = widgets.Button(description="⏹️ Stop Training", button_style='danger', disabled=True,
                                layout=widgets.Layout(width='140px', height='35px'))
    reset_button = widgets.Button(description="🔄 Reset Metrics", button_style='warning',
                                 layout=widgets.Layout(width='140px', height='35px'))
    validate_button = widgets.Button(description="🔍 Validate Model", button_style='info',
                                    layout=widgets.Layout(width='140px', height='35px'))
    
    # Container layout
    button_container = widgets.HBox([
        primary_buttons['download_button'], stop_button, reset_button, validate_button
    ], layout=widgets.Layout(margin='5px 0', justify_content='flex-start'))
    
    return {
        'start_button': primary_buttons['download_button'],
        'stop_button': stop_button,
        'reset_button': reset_button,
        'validate_button': validate_button,
        'button_container': button_container
    }

def _create_yaml_config_tabs(config: Dict[str, Any]) -> widgets.Tab:
    """Create config tabs dengan YAML config integration"""
    from smartcash.ui.components.tab_factory import create_tab_widget
    
    # Extract YAML configs
    model_config = config.get('model', {})
    training_config = config.get('training', {}) 
    hyperparams = config.get('hyperparameters', {})
    training_utils = config.get('training_utils', {})
    
    # Refresh button untuk config update
    refresh_button = widgets.Button(description="🔄", button_style='info', tooltip='Refresh dari YAML',
                                   layout=widgets.Layout(width='50px', height='28px'))
    
    # Tab 1: Model Configuration (dari model_config.yaml)
    model_html = f"""
    <div style="padding: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h5 style="margin: 0; color: #1976d2;">🧠 Model Configuration</h5>
        </div>
        <ul style="font-size: 13px; margin: 5px 0; padding-left: 20px;">
            <li><b>Type:</b> <span style="color: #1976d2;">{model_config.get('type', 'efficient_basic')}</span></li>
            <li><b>Backbone:</b> <span style="color: #1976d2;">{model_config.get('backbone', 'efficientnet_b4')}</span></li>
            <li><b>Pretrained:</b> <span style="color: {'#2e7d32' if model_config.get('backbone_pretrained', True) else '#d32f2f'};">{'✅' if model_config.get('backbone_pretrained', True) else '❌'}</span></li>
            <li><b>Confidence:</b> <span style="color: #1976d2;">{model_config.get('confidence', 0.25)}</span></li>
            <li><b>IoU Threshold:</b> <span style="color: #1976d2;">{model_config.get('iou_threshold', 0.45)}</span></li>
            <li><b>Optimizations:</b> 
                <span style="color: {'#2e7d32' if model_config.get('use_attention') else '#666'};">Attention: {'✅' if model_config.get('use_attention') else '❌'}</span>,
                <span style="color: {'#2e7d32' if model_config.get('use_residual') else '#666'};">Residual: {'✅' if model_config.get('use_residual') else '❌'}</span>,
                <span style="color: {'#2e7d32' if model_config.get('use_ciou') else '#666'};">CIoU: {'✅' if model_config.get('use_ciou') else '❌'}</span>
            </li>
        </ul>
    </div>
    """
    
    # Tab 2: Training Parameters (dari training_config.yaml dan hyperparameters_config.yaml)
    training_html = f"""
    <div style="padding: 10px;">
        <h5 style="margin: 0 0 10px 0; color: #6a1b9a;">⚙️ Training Parameters</h5>
        <ul style="font-size: 13px; margin: 5px 0; padding-left: 20px;">
            <li><b>Epochs:</b> <span style="color: #6a1b9a;">{config.get('epochs', 100)}</span></li>
            <li><b>Batch Size:</b> <span style="color: #6a1b9a;">{config.get('batch_size', 16)}</span></li>
            <li><b>Learning Rate:</b> <span style="color: #6a1b9a;">{config.get('learning_rate', 0.001)}</span></li>
            <li><b>Optimizer:</b> <span style="color: #6a1b9a;">{config.get('optimizer', 'Adam')}</span></li>
            <li><b>Scheduler:</b> <span style="color: #6a1b9a;">{config.get('scheduler', 'cosine')}</span></li>
            <li><b>Weight Decay:</b> <span style="color: #6a1b9a;">{config.get('weight_decay', 0.0005)}</span></li>
            <li><b>Layer Mode:</b> <span style="color: #6a1b9a;">{training_utils.get('layer_mode', 'single')}</span></li>
        </ul>
    </div>
    """
    
    # Tab 3: Training Strategy & Utils
    strategy_html = f"""
    <div style="padding: 10px;">
        <h5 style="margin: 0 0 10px 0; color: #2e7d32;">🚀 Training Strategy</h5>
        <ul style="font-size: 13px; margin: 5px 0; padding-left: 20px;">
            <li><b>Early Stopping:</b> <span style="color: {'#2e7d32' if config.get('early_stopping', {}).get('enabled', True) else '#d32f2f'};">{'✅ Enabled' if config.get('early_stopping', {}).get('enabled', True) else '❌ Disabled'}</span> (Patience: {config.get('early_stopping', {}).get('patience', 15)})</li>
            <li><b>Save Best:</b> <span style="color: {'#2e7d32' if config.get('save_best', {}).get('enabled', True) else '#d32f2f'};">{'✅ Enabled' if config.get('save_best', {}).get('enabled', True) else '❌ Disabled'}</span></li>
            <li><b>Mixed Precision:</b> <span style="color: {'#2e7d32' if training_utils.get('mixed_precision', True) else '#d32f2f'};">{'✅ Enabled' if training_utils.get('mixed_precision', True) else '❌ Disabled'}</span></li>
            <li><b>Multi-Scale:</b> <span style="color: {'#2e7d32' if config.get('multi_scale', True) else '#d32f2f'};">{'✅ Enabled' if config.get('multi_scale', True) else '❌ Disabled'}</span></li>
            <li><b>Experiment:</b> <span style="color: #2e7d32;">{training_utils.get('experiment_name', 'efficientnet_b4_training')}</span></li>
        </ul>
    </div>
    """
    
    # Tab 4: Paths & Validation
    paths_html = f"""
    <div style="padding: 10px;">
        <h5 style="margin: 0 0 10px 0; color: #00695c;">📁 Paths & Validation</h5>
        <ul style="font-size: 13px; margin: 5px 0; padding-left: 20px;">
            <li><b>Checkpoint Dir:</b> <span style="color: #00695c; font-family: monospace;">{training_utils.get('checkpoint_dir', '/content/runs/train/checkpoints')}</span></li>
            <li><b>Pretrained Models:</b> <span style="color: #00695c; font-family: monospace;">{config.get('pretrained_models_path', '/content/drive/MyDrive/SmartCash/models')}</span></li>
            <li><b>Validation Frequency:</b> <span style="color: #00695c;">{config.get('validation', {}).get('frequency', 1)} epoch(s)</span></li>
            <li><b>IoU Threshold:</b> <span style="color: #00695c;">{config.get('validation', {}).get('iou_thres', 0.6)}</span></li>
            <li><b>Conf Threshold:</b> <span style="color: #00695c;">{config.get('validation', {}).get('conf_thres', 0.001)}</span></li>
            <li><b>Log Metrics Every:</b> <span style="color: #00695c;">{training_utils.get('log_metrics_every', 10)} batches</span></li>
        </ul>
    </div>
    """
    
    # Create tabs dengan refresh button di model tab
    model_widget = widgets.VBox([
        widgets.HBox([refresh_button], layout=widgets.Layout(justify_content='flex-end')),
        widgets.HTML(model_html)
    ])
    
    tabs = create_tab_widget([
        ('Model', model_widget),
        ('Training', widgets.HTML(training_html)),
        ('Strategy', widgets.HTML(strategy_html)),
        ('Paths', widgets.HTML(paths_html))
    ])
    
    # Store refresh button reference
    setattr(tabs, '_refresh_button', refresh_button)
    return tabs

def _create_simple_fallback(error_msg: str) -> Dict[str, Any]:
    """Simple fallback untuk error cases tanpa excessive components"""
    from smartcash.ui.utils.fallback_utils import create_fallback_ui
    
    fallback = create_fallback_ui(f"Training form error: {error_msg}", 'training')
    
    # Add minimal required components
    fallback.update({
        'start_button': widgets.Button(description="🚀 Training", button_style='success', disabled=True),
        'stop_button': widgets.Button(description="⏹️ Stop", disabled=True),
        'reset_button': widgets.Button(description="🔄 Reset", disabled=True),
        'log_output': widgets.Output(),
        'chart_output': widgets.Output(),
        'metrics_output': widgets.Output(),
        'config': {}
    })
    
    return fallback

def update_config_tabs_in_form(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update config tabs dengan YAML config baru"""
    config_tabs = ui_components.get('config_tabs')
    if not config_tabs:
        return
    
    try:
        # Create new tabs dengan updated config
        updated_tabs = _create_yaml_config_tabs(config)
        
        # Preserve selected index
        selected_index = getattr(config_tabs, 'selected_index', 0)
        
        # Update children dan titles
        config_tabs.children = updated_tabs.children
        [config_tabs.set_title(i, updated_tabs.titles[i]) for i in range(len(updated_tabs.children))]
        
        # Restore selected index
        config_tabs.selected_index = selected_index
        
        # Update refresh button reference
        refresh_button = getattr(updated_tabs, '_refresh_button', None)
        refresh_button and setattr(config_tabs, '_refresh_button', refresh_button)
        
        # Update UI components reference
        ui_components['config_tabs'] = config_tabs
        ui_components['info_display'] = config_tabs
        ui_components['refresh_button'] = refresh_button
        
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.warning(f"⚠️ Error updating config tabs: {str(e)}")

# One-liner utilities untuk form management
get_refresh_button_from_tabs = lambda tabs: getattr(tabs, '_refresh_button', None)
extract_yaml_model_info = lambda config: f"{config.get('model', {}).get('backbone', 'efficientnet_b4')} ({config.get('model', {}).get('type', 'efficient_basic')})"
get_training_summary = lambda config: f"{config.get('epochs', 100)} epochs, {config.get('batch_size', 16)} batch, {config.get('optimizer', 'Adam')} optimizer"