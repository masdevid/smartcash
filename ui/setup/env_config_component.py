"""
File: smartcash/ui/setup/env_config_component.py
Deskripsi: Komponen UI untuk konfigurasi environment dengan konsolidasi reusable components
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_env_config_ui(env, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat komponen UI untuk environment config dengan styling standar.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI standar
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.info_boxes.environment_info import get_environment_info
    
    # Reuse button components dari ui_helpers jika tersedia
    try:
        from smartcash.ui.components.action_buttons import create_action_buttons
        has_action_buttons = True
    except ImportError:
        has_action_buttons = False
    
    # Header
    header = create_header(
        "⚙️ Konfigurasi Environment", 
        "Setup environment untuk project SmartCash"
    )
    
    # Progress tracker
    progress_components = _create_progress_components()
    progress_bar = progress_components['progress_bar']
    progress_message = progress_components['progress_message']
    progress_container = progress_components['container']
    
    # Panel status Colab/Drive
    colab_panel = widgets.HTML(value="Mendeteksi environment...")
    
    # Button components (reuse atau create)
    if has_action_buttons:
        # Reuse dari komponen action_buttons
        buttons = create_action_buttons(
            primary_label="Hubungkan Google Drive",
            primary_icon="link",
            secondary_buttons=[("Setup Direktori Lokal", "folder-plus", "info")],
            cleanup_enabled=False
        )
        drive_button = buttons['primary_button']
        directory_button = buttons['secondary_buttons'][0]
        button_container = buttons['container']
    else:
        # Create manual jika komponen tidak tersedia
        drive_button = widgets.Button(
            description="Hubungkan Google Drive",
            button_style="primary",
            icon="link",
            layout=widgets.Layout(margin='5px')
        )
        
        directory_button = widgets.Button(
            description="Setup Direktori Lokal",
            button_style="info",
            icon="folder-plus",
            layout=widgets.Layout(margin='5px')
        )
        
        button_container = widgets.HBox(
            [drive_button, directory_button],
            layout=widgets.Layout(
                display='flex',
                flex_flow='row wrap',
                justify_content='flex-start',
                gap='10px',
                margin='10px 0'
            )
        )
    
    # Panel log messages
    log_output = _create_log_output()
    
    # Panel info bantuan
    help_box = get_environment_info()
    
    # Divider
    divider = widgets.HTML(f"<hr style='margin: 15px 0; border: 0; border-top: 1px solid {COLORS['border']};'>")
    
    # Container utama
    ui = widgets.VBox([
        header,
        colab_panel,
        progress_container,
        divider, 
        button_container,
        log_output,
        help_box
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Komponen UI dengan referensi ke status untuk logging
    ui_components = {
        'ui': ui,
        'header': header,
        'colab_panel': colab_panel,
        'drive_button': drive_button,
        'directory_button': directory_button,
        'progress_bar': progress_bar,
        'progress_message': progress_message,
        'progress_container': progress_container,
        'status': log_output['output'],
        'log_box': log_output['container'],
        'help_panel': help_box,
        'module_name': 'env_config'
    }
    
    return ui_components

def _create_progress_components() -> Dict[str, widgets.Widget]:
    """
    Buat komponen progress tracker.
    
    Returns:
        Dictionary berisi komponen progress
    """
    # Progress bar
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=5,
        description='Progress:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%', visibility='hidden')
    )
    
    # Progress message
    progress_message = widgets.HTML(
        value="Siap digunakan",
        layout=widgets.Layout(margin='0 0 0 10px', padding='5px', visibility='hidden')
    )
    
    # Container untuk progress components
    progress_container = widgets.HBox(
        [progress_bar, progress_message], 
        layout=widgets.Layout(margin='10px 0', align_items='center')
    )
    
    return {
        'progress_bar': progress_bar,
        'progress_message': progress_message,
        'container': progress_container
    }

def _create_log_output() -> Dict[str, widgets.Widget]:
    """
    Buat komponen output untuk log.
    
    Returns:
        Dictionary berisi komponen log output
    """
    from smartcash.ui.utils.constants import COLORS
    
    # Header untuk log
    log_header = widgets.HTML(f"<h3 style='margin: 10px 0 5px 0; color: {COLORS['dark']};'>{COLORS['info']} Log Messages</h3>")
    
    # Output widget
    output = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border=f'1px solid {COLORS["primary"]}',
            min_height='150px',
            max_height='300px',
            margin='0 0 10px 0',
            padding='10px',
            overflow='auto',
            background_color=f'{COLORS["light"]}'
        )
    )
    
    # Container untuk log components
    container = widgets.VBox([
        log_header,
        output
    ], layout=widgets.Layout(margin='15px 0'))
    
    return {
        'header': log_header,
        'output': output,
        'container': container
    }