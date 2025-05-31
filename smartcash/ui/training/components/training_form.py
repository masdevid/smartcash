"""\nFile: smartcash/ui/training/components/training_form.py\nDeskripsi: Training form components dengan integration SRP files yang lebih kecil\n"""

import ipywidgets as widgets
from typing import Dict, Any


def create_training_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create training form components dengan proper fallback handling"""
    try:
        from smartcash.ui.components.progress_tracking import create_progress_tracking_container
        from smartcash.ui.components.log_accordion import create_log_accordion
        from smartcash.ui.components.status_panel import create_status_panel
        from smartcash.ui.training.components.control_buttons import create_training_control_buttons
        from smartcash.ui.training.components.config_tabs import create_config_tabs
        from smartcash.ui.training.components.metrics_accordion import create_metrics_accordion
        
        # Training control buttons
        control_buttons = create_training_control_buttons()
        
        # Progress tracking dengan proper container
        progress_components = create_progress_tracking_container()
        
        # Log components dengan existing log_accordion
        log_components = create_log_accordion('training', height='250px')
        
        # Status panel untuk training feedback
        status_panel = create_status_panel("🧠 EfficientNet-B4 siap untuk training", "info")
        
        # Config tabs untuk configuration summary
        config_tabs = create_config_tabs(config)
        
        # Metrics accordion untuk chart dan metrics
        metrics_components = create_metrics_accordion()
        
        return {
            # Control buttons
            **control_buttons,
            'control_buttons': control_buttons,  # Explicit key for container access
            
            # Progress tracking
            'progress_container': progress_components.get('container'),
            'progress_tracker': progress_components.get('tracker'),
            'status_panel': status_panel,
            
            # Config tabs
            'config_tabs': config_tabs,
            'info_display': config_tabs,  # For backward compatibility
            
            # Metrics components
            **metrics_components,
            
            # Log components
            'log_output': log_components.get('log_output'),
            'log_accordion': log_components.get('log_accordion'),
            
            # Config reference
            'config': config.get('training', {})
        }
        
    except Exception as e:
        # Simple fallback untuk prevent complete failure
        from smartcash.ui.training.components.fallback_component import create_fallback_training_form
        return create_fallback_training_form(str(e))


# One-liner utilities untuk update config tabs dengan proper title handling
def update_config_tabs(tabs_widget, config):
    """Update tab widget dengan config baru dan mempertahankan titles"""
    new_tabs = create_config_tabs(config)
    # Copy children dan update titles
    tabs_widget.children = new_tabs.children
    # Pastikan titles diperbarui dengan benar
    for i in range(len(tabs_widget.children)):
        if i < len(new_tabs.titles):
            tabs_widget.set_title(i, new_tabs.titles[i])
    return tabs_widget