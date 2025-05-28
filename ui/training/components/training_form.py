"""
File: smartcash/ui/training/components/training_form.py
Deskripsi: Form components untuk training configuration menggunakan shared components
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.constants import ICONS, COLORS


def create_training_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create training form components dengan shared components"""
    from smartcash.ui.components.action_buttons import create_action_buttons
    from smartcash.ui.components.progress_tracking import create_progress_tracking_container
    from smartcash.ui.components.log_accordion import create_log_accordion
    from smartcash.ui.components.status_panel import create_status_panel
    
    training_config = config.get('training', {})
    
    # Training control buttons - one-liner setup
    action_buttons = create_action_buttons(
        primary_label="🚀 Mulai Training", primary_icon="play", cleanup_enabled=True,
        secondary_buttons=[("⏹️ Stop", "stop", "danger"), ("🔄 Reset", "refresh", "warning")]
    )
    
    # Progress tracking components
    progress_components = create_progress_tracking_container()
    
    # Log accordion untuk training logs
    log_components = create_log_accordion('training', height='250px')
    
    # Status panel untuk feedback
    status_panel = create_status_panel("Siap untuk memulai training", "info")
    
    # Training info display - one-liner HTML component
    info_display = widgets.Output(layout=widgets.Layout(
        width='100%', border='1px solid #ddd', padding='10px', 
        margin='10px 0', max_height='200px', overflow_y='auto'
    ))
    
    # Metrics visualization output
    metrics_output = widgets.Output(layout=widgets.Layout(
        width='100%', max_height='300px', height='auto', 
        margin='10px 0', border='1px solid #ddd', 
        padding='10px', overflow_y='auto'
    ))
    
    # Chart output for training metrics
    chart_output = widgets.Output(layout=widgets.Layout(
        width='100%', max_height='400px', margin='10px 0', 
        border='1px solid #ddd', padding='10px', overflow_y='auto'
    ))
    
    return {
        # Action buttons
        'start_button': action_buttons['download_button'],  # Primary button
        'stop_button': create_stop_button(),
        'reset_button': create_reset_button(),
        'cleanup_button': action_buttons.get('cleanup_button'),
        'button_container': action_buttons['container'],
        
        # Progress and status
        'progress_container': progress_components['container'],
        'progress_bar': progress_components.get('tracker'),
        'status_panel': status_panel,
        
        # Outputs
        'log_output': log_components['log_output'],
        'log_accordion': log_components['log_accordion'],
        'info_display': info_display,
        'metrics_output': metrics_output,
        'chart_output': chart_output,
        
        # Config data
        'config': training_config
    }


def create_stop_button() -> widgets.Button:
    """Create stop button dengan consistent styling"""
    return widgets.Button(
        description="⏹️ Stop Training", button_style='danger', 
        tooltip='Hentikan proses training', disabled=True,
        layout=widgets.Layout(width='140px', height='35px')
    )


def create_reset_button() -> widgets.Button:
    """Create reset button untuk reset metrics"""
    return widgets.Button(
        description="🔄 Reset Metrics", button_style='warning',
        tooltip='Reset chart dan metrics', 
        layout=widgets.Layout(width='140px', height='35px')
    )