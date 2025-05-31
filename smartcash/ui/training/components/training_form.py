"""
File: smartcash/ui/training/components/training_form.py
Deskripsi: Fixed training form components dengan proper error handling dan progress integration
"""

import ipywidgets as widgets
from typing import Dict, Any


def create_training_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create training form components dengan proper fallback handling"""
    try:
        from smartcash.ui.components.progress_tracking import create_progress_tracking_container
        from smartcash.ui.components.log_accordion import create_log_accordion
        from smartcash.ui.components.status_panel import create_status_panel
        
        # Training control buttons
        control_buttons = create_training_control_buttons()
        
        # Progress tracking dengan proper container
        progress_components = create_progress_tracking_container()
        
        # Log components dengan existing log_accordion
        log_components = create_log_accordion('training', height='250px')
        
        # Status panel untuk training feedback
        status_panel = create_status_panel("🧠 EfficientNet-B4 siap untuk training", "info")
        
        # Critical display components dengan safe layout
        display_components = create_display_components()
        
        # Info display untuk configuration summary
        info_display = create_info_display(config)
        
        return {
            # Control buttons
            **control_buttons,
            
            # Progress tracking
            'progress_container': progress_components.get('container'),
            'progress_tracker': progress_components.get('tracker'),
            'status_panel': status_panel,
            
            # Display components
            **display_components,
            
            # Log components
            'log_output': log_components.get('log_output'),
            'log_accordion': log_components.get('log_accordion'),
            
            # Info display
            'info_display': info_display,
            
            # Config reference
            'config': config.get('training', {})
        }
        
    except Exception as e:
        # Simple fallback untuk prevent complete failure
        return create_fallback_training_form(str(e))


def create_training_control_buttons() -> Dict[str, Any]:
    """Create training control buttons using existing action_buttons"""
    from smartcash.ui.components.action_buttons import create_action_buttons
    
    # Primary button (Start Training)
    primary_buttons = create_action_buttons(
        primary_label="🚀 Mulai Training",
        primary_icon="",
        primary_style='success',
        secondary_buttons=[],
        cleanup_enabled=False,
        button_width='160px'
    )
    
    # Secondary control buttons
    stop_button = widgets.Button(
        description="⏹️ Stop Training",
        button_style='danger',
        tooltip='Hentikan training dan simpan checkpoint',
        disabled=True,
        layout=widgets.Layout(width='140px', height='35px', margin='2px')
    )
    
    reset_button = widgets.Button(
        description="🔄 Reset Metrics", 
        button_style='warning',
        tooltip='Reset training metrics dan chart',
        layout=widgets.Layout(width='140px', height='35px', margin='2px')
    )
    
    validate_button = widgets.Button(
        description="🔍 Cek Model",
        button_style='info', 
        tooltip='Validasi model readiness',
        layout=widgets.Layout(width='130px', height='35px', margin='2px')
    )
    
    cleanup_button = widgets.Button(
        description="🧹 Cleanup GPU",
        button_style='',
        tooltip='Bersihkan GPU memory',
        layout=widgets.Layout(width='130px', height='35px', margin='2px')
    )
    
    # Container layout
    buttons_row_1 = widgets.HBox([primary_buttons['download_button'], stop_button, reset_button])
    buttons_row_2 = widgets.HBox([validate_button, cleanup_button])
    
    button_container = widgets.VBox([
        buttons_row_1, buttons_row_2
    ], layout=widgets.Layout(
        width='100%', align_items='center', border='1px solid #dee2e6',
        border_radius='8px', padding='15px', margin='10px 0',
        background_color='#f8f9fa'
    ))
    
    return {
        'start_button': primary_buttons['download_button'],  # Map to start_button
        'stop_button': stop_button,
        'reset_button': reset_button, 
        'validate_button': validate_button,
        'cleanup_button': cleanup_button,
        'button_container': button_container
    }


def create_display_components() -> Dict[str, Any]:
    """Create display components dengan proper styling"""
    
    # Model readiness display
    model_readiness_display = widgets.Output(layout=widgets.Layout(
        width='100%', border='1px solid #28a745', padding='12px',
        margin='10px 0', max_height='200px', overflow_y='auto',
        background_color='#f8fff8', border_radius='6px'
    ))
    
    # Training config display
    training_config_display = widgets.Output(layout=widgets.Layout(
        width='100%', border='1px solid #007bff', padding='12px',
        margin='10px 0', max_height='180px', overflow_y='auto',
        background_color='#f8f9ff', border_radius='6px'
    ))
    
    # GPU status display
    gpu_status_display = widgets.Output(layout=widgets.Layout(
        width='100%', border='1px solid #ffc107', padding='12px',
        margin='10px 0', max_height='180px', overflow_y='auto',
        background_color='#fffef8', border_radius='6px'
    ))
    
    # Metrics display
    metrics_output = widgets.Output(layout=widgets.Layout(
        width='100%', max_height='280px', margin='10px 0',
        border='1px solid #17a2b8', padding='12px',
        overflow_y='auto', border_radius='6px'
    ))
    
    # Chart display
    chart_output = widgets.Output(layout=widgets.Layout(
        width='100%', max_height='420px', margin='10px 0',
        border='1px solid #6f42c1', padding='12px',
        overflow_y='auto', border_radius='6px'
    ))
    
    return {
        'model_readiness_display': model_readiness_display,
        'training_config_display': training_config_display,
        'gpu_status_display': gpu_status_display,
        'metrics_output': metrics_output,
        'chart_output': chart_output
    }


def create_info_display(config: Dict[str, Any]) -> widgets.HTML:
    """Create info display untuk configuration summary"""
    training_config = config.get('training', {})
    model_type = training_config.get('model_type', 'efficient_optimized')
    
    info_html = f"""
    <div style="padding: 12px; background-color: #f8f9fa; border-radius: 6px;">
        <h5>🧠 Model Configuration</h5>
        <ul style="margin: 10px 0;">
            <li><b>Model Type:</b> {model_type}</li>
            <li><b>Backbone:</b> {training_config.get('backbone', 'efficientnet_b4')}</li>
            <li><b>Epochs:</b> {training_config.get('epochs', 100)}</li>
            <li><b>Batch Size:</b> {training_config.get('batch_size', 16)}</li>
            <li><b>Learning Rate:</b> {training_config.get('learning_rate', 0.001)}</li>
        </ul>
    </div>
    """
    
    return widgets.HTML(value=info_html)


def create_fallback_training_form(error_msg: str) -> Dict[str, Any]:
    """Simple fallback training form untuk error cases"""
    
    # Basic buttons dengan minimal functionality
    start_button = widgets.Button(description="🚀 Mulai Training", button_style='success')
    stop_button = widgets.Button(description="⏹️ Stop", button_style='danger', disabled=True)
    reset_button = widgets.Button(description="🔄 Reset", button_style='warning')
    
    # Basic container
    button_container = widgets.HBox([start_button, stop_button, reset_button])
    
    # Error status
    status_panel = widgets.HTML(f"""
    <div style="padding: 10px; background: #ffeaea; border-radius: 6px; color: #d32f2f;">
        ❌ Error creating training form: {error_msg}
    </div>
    """)
    
    # Basic outputs
    log_output = widgets.Output()
    progress_container = widgets.VBox([])
    
    return {
        'start_button': start_button,
        'stop_button': stop_button,
        'reset_button': reset_button,
        'button_container': button_container,
        'status_panel': status_panel,
        'log_output': log_output,
        'progress_container': progress_container,
        'info_display': widgets.HTML("Training form error"),
        'model_readiness_display': widgets.Output(),
        'training_config_display': widgets.Output(),
        'gpu_status_display': widgets.Output(),
        'metrics_output': widgets.Output(),
        'chart_output': widgets.Output(),
        'config': {}
    }


# One-liner utilities
update_info_display = lambda info_widget, config: setattr(info_widget, 'value', create_info_display(config).value)
enable_training_mode = lambda ui: [setattr(ui.get(btn), 'disabled', False) for btn in ['start_button', 'reset_button']] and setattr(ui.get('stop_button'), 'disabled', True)
enable_stopping_mode = lambda ui: setattr(ui.get('start_button'), 'disabled', True) or setattr(ui.get('stop_button'), 'disabled', False)