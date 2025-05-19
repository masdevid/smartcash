"""
File: smartcash/ui/setup/env_config/components/ui_creator.py
Deskripsi: Creator untuk komponen UI environment config
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.utils.alert_utils import create_info_box
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS

def create_env_config_ui() -> Dict[str, Any]:
    """
    Buat komponen UI untuk konfigurasi environment
    
    Returns:
        Dictionary berisi komponen UI
    """
    # Header dengan komponen standar
    header = create_header(
        "⚙️ Environment Configuration", 
        "Setup environment dan konfigurasi SmartCash"
    )
    
    # Status panel menggunakan komponen alert standar
    status_panel = widgets.HTML(
        create_info_box(
            "Environment Status", 
            "Connect to Google Drive dan setup directory",
            style="info"
        ).value
    )
    
    # Action buttons in HBox
    drive_button = widgets.Button(
        description="Connect Drive",
        button_style="success",
        icon="cloud-upload",
        tooltip="Connect to Google Drive",
        layout=widgets.Layout(width='auto', margin='0 5px')
    )
    
    directory_button = widgets.Button(
        description="Setup Directory",
        button_style="warning",
        icon="folder",
        tooltip="Setup directory structure",
        layout=widgets.Layout(width='auto', margin='0 5px')
    )
    
    action_buttons = widgets.HBox(
        [drive_button, directory_button],
        layout=widgets.Layout(
            display='flex',
            flex_flow='row',
            justify_content='center',
            width='100%',
            margin='10px 0'
        )
    )
    
    # Progress section
    progress = widgets.FloatProgress(
        value=0.0,
        min=0,
        max=1.0,
        description='Progress:',
        bar_style='info',
        style={'description_width': 'initial'},
        layout=widgets.Layout(
            width='100%',
            margin='10px 0',
            visibility='hidden'  # Hidden by default
        )
    )
    
    # Progress label
    progress_label = widgets.HTML(
        value="",
        layout=widgets.Layout(
            margin='5px 0',
            visibility='hidden'  # Hidden by default
        )
    )
    
    # Log section
    log = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border=f'1px solid {COLORS["border"]}',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            padding='10px',
            overflow='auto'
        )
    )
    
    # Container utama dengan semua komponen
    main = widgets.VBox(
        [
            header,
            status_panel,
            action_buttons,
            progress,
            progress_label,
            log
        ],
        layout=widgets.Layout(
            width='100%',
            padding='10px'
        )
    )
    
    # Struktur final komponen UI
    ui_components = {
        'ui': main,
        'status_panel': status_panel,
        'drive_button': drive_button,
        'directory_button': directory_button,
        'progress': progress,
        'progress_label': progress_label,
        'log': log,
        'module_name': 'env_config'
    }
    
    return ui_components 