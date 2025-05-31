"""
File: smartcash/ui/training/components/control_buttons.py
Deskripsi: Komponen button kontrol untuk training UI
"""

import ipywidgets as widgets
from typing import Dict, Any, List

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
    
    refresh_button = widgets.Button(
        description="🔄 Refresh Config",
        button_style='info',
        tooltip='Perbarui informasi konfigurasi dari modul lain',
        layout=widgets.Layout(width='130px', height='35px', margin='2px')
    )
    
    cleanup_button = widgets.Button(
        description="🧹 Cleanup GPU",
        button_style='',
        tooltip='Bersihkan GPU memory',
        layout=widgets.Layout(width='130px', height='35px', margin='2px')
    )
    
    # Container layout - Semua button dalam satu baris
    button_container = widgets.HBox([
        primary_buttons['download_button'], 
        stop_button, 
        reset_button,
        validate_button, 
        cleanup_button,
        refresh_button
    ], layout=widgets.Layout(
        margin='5px 0',
        padding='5px',
        width='100%',
        justify_content='flex-start',
        align_items='center'
    ))
    
    return {
        'start_button': primary_buttons['download_button'],
        'stop_button': stop_button,
        'reset_button': reset_button,
        'validate_button': validate_button,
        'cleanup_button': cleanup_button,
        'refresh_button': refresh_button,
        'button_container': button_container
    }

# One-liner utilities
enable_training_mode = lambda ui: [setattr(ui.get(btn), 'disabled', False) for btn in ['start_button', 'reset_button']] and setattr(ui.get('stop_button'), 'disabled', True)
enable_stopping_mode = lambda ui: setattr(ui.get('start_button'), 'disabled', True) or setattr(ui.get('stop_button'), 'disabled', False)
