"""
File: smartcash/ui/training/components/training_form.py
Deskripsi: Fixed training form tanpa refresh button di dalam tab content
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_training_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create training form dengan YAML config integration - refresh button external"""
    try:
        from smartcash.ui.components.progress_tracker import create_three_progress_tracker
        from smartcash.ui.components.log_accordion import create_log_accordion
        from smartcash.ui.components.status_panel import create_status_panel
        from smartcash.ui.training.components.config_tabs import create_config_tabs
        
        # Control buttons dengan YAML-based model info
        control_buttons = _create_control_buttons(config)
        
        # Progress tracking dengan three level untuk training epochs, batches, dan metrics
        # Menggunakan steps yang sesuai dengan proses training
        progress_components = create_three_progress_tracker(
            operation="Training",
            steps=["Initialization", "Training", "Validation", "Checkpoint"]
        )
        
        # Status panel dengan model-specific message
        model_type = config.get('model', {}).get('type', 'efficient_basic')
        backbone = config.get('model', {}).get('backbone', 'efficientnet_b4')
        status_message = f"🧠 {backbone.upper()} ({model_type}) siap untuk training"
        status_panel = create_status_panel(status_message, "info")
        
        # Log accordion dengan training namespace
        log_components = create_log_accordion('training', height='250px')
        
        # Config tabs tanpa refresh button internal - dipindah ke layout
        config_tabs = create_config_tabs(config)
        
        # Chart & metrics outputs
        chart_output = widgets.Output(layout=widgets.Layout(max_height='400px', overflow='auto'))
        metrics_output = widgets.Output(layout=widgets.Layout(max_height='150px', overflow='auto'))
        
        # Refresh button terpisah untuk external placement
        refresh_button = widgets.Button(
            description="🔄 Refresh Config",
            button_style='info',
            tooltip='Refresh konfigurasi dari YAML files',
            layout=widgets.Layout(width='140px', height='30px')
        )
        
        # Memastikan semua komponen yang diperlukan tersedia
        return {
            **control_buttons,
            'progress_container': progress_components['container'],
            'progress_tracker': progress_components['tracker'],
            'update_progress': progress_components['update_progress'],
            'update_overall': progress_components['update_overall'],
            'update_step': progress_components['update_step'],
            'update_current': progress_components['update_current'],
            'show_container': progress_components['show_container'],
            'hide_container': progress_components['hide_container'],
            'show_for_operation': progress_components['show_for_operation'],
            'complete_operation': progress_components['complete_operation'],
            'error_operation': progress_components['error_operation'],
            'reset_all': progress_components['reset_all'],
            'status_panel': status_panel,
            'config_tabs': config_tabs,  # Pastikan config_tabs dimasukkan ke dalam dictionary
            'info_display': config_tabs,
            'log_output': log_components.get('log_output'),
            'log_accordion': log_components.get('log_accordion'),
            'chart_output': chart_output,
            'metrics_output': metrics_output,
            'refresh_button': refresh_button,  # External button
            'container': widgets.VBox([  # Tambahkan container untuk komponen UI
                status_panel,
                config_tabs,
                control_buttons['button_container']
            ]),
            'config': config
        }
        
    except Exception as e:
        return _create_simple_fallback(str(e))

def _create_control_buttons(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create enhanced control buttons dengan YAML config awareness"""
    from smartcash.ui.components.action_buttons import create_action_buttons
    
    # Extract info dari YAML config untuk button labels
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
    
    # Control buttons dengan consistent sizing
    button_layout = widgets.Layout(width='140px', height='35px')
    
    stop_button = widgets.Button(description="⏹️ Stop Training", button_style='danger', disabled=True, layout=button_layout)
    reset_button = widgets.Button(description="🔄 Reset Metrics", button_style='warning', layout=button_layout)
    validate_button = widgets.Button(description="🔍 Validate Model", button_style='info', layout=button_layout)
    
    # Container layout dengan proper spacing
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

def _create_simple_fallback(error_msg: str) -> Dict[str, Any]:
    """Simple fallback untuk error cases"""
    from smartcash.ui.utils.fallback_utils import create_fallback_ui
    
    fallback = create_fallback_ui(f"Training form error: {error_msg}", 'training')
    
    # Add minimal required components dengan one-liner initialization
    required_components = {
        'start_button': widgets.Button(description="🚀 Training", button_style='success', disabled=True),
        'stop_button': widgets.Button(description="⏹️ Stop", disabled=True),
        'reset_button': widgets.Button(description="🔄 Reset", disabled=True),
        'validate_button': widgets.Button(description="🔍 Validate", disabled=True),
        'refresh_button': widgets.Button(description="🔄 Refresh", disabled=True),
        'log_output': widgets.Output(),
        'chart_output': widgets.Output(),
        'metrics_output': widgets.Output(),
        'config': {}
    }
    
    fallback.update(required_components)
    return fallback

def update_config_tabs_in_form(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update config tabs dengan YAML config baru - fixed untuk proper tab updating"""
    config_tabs = ui_components.get('config_tabs')
    if not config_tabs:
        return
    
    try:
        from smartcash.ui.training.components.config_tabs import update_config_tabs
        
        # Update tabs dengan new config
        updated_tabs = update_config_tabs(config_tabs, config)
        
        # Update UI components reference
        ui_components['config_tabs'] = updated_tabs
        ui_components['info_display'] = updated_tabs
        
        # Log successful update
        logger = ui_components.get('logger')
        if logger:
            logger.info("🔄 Config tabs berhasil diperbarui dari YAML")
        
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"⚠️ Error updating config tabs: {str(e)}")

# One-liner utilities untuk form management
get_model_summary = lambda config: f"{config.get('model', {}).get('backbone', 'efficientnet_b4')} ({config.get('model', {}).get('type', 'efficient_basic')})"
get_training_summary = lambda config: f"{config.get('epochs', 100)} epochs, batch {config.get('batch_size', 16)}, {config.get('optimizer', 'Adam')} optimizer"
extract_button_config = lambda config, model_type: f"🚀 Train {model_type.title()} ({config.get('epochs', 100)} epochs)"