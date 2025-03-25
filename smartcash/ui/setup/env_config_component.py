"""
File: smartcash/ui/setup/env_config_component.py
Deskripsi: Komponen UI untuk konfigurasi environment dengan struktur yang lebih modular dan terorganisir
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_env_config_ui(env, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat komponen UI untuk konfigurasi environment.
    
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
    
    # Header
    header = create_header(
        "⚙️ Konfigurasi Environment", 
        "Setup environment sistem untuk SmartCash"
    )
    
    # Panel Colab Info
    colab_panel = widgets.HTML(
        """<div style="padding: 10px; background-color:#d1ecf1; color:#0c5460; 
           border-radius:4px; margin:10px 0; border-left:4px solid #0c5460;">
           <p style="margin:5px 0">ℹ️ Mendeteksi environment...</p>
           </div>"""
    )
    
    # Tombol Connect Drive (untuk Colab)
    drive_button = widgets.Button(
        description='Hubungkan Google Drive',
        button_style='info',
        icon='link',
        tooltip="Hubungkan Google Drive untuk penyimpanan dataset dan konfigurasi",
        layout=widgets.Layout(margin='5px', display='none')  # Hidden by default
    )
    
    # Tombol Setup Direktori
    directory_button = widgets.Button(
        description='Setup Direktori Lokal',
        button_style='primary',
        icon='folder-plus',
        tooltip="Buat struktur direktori untuk SmartCash",
        layout=widgets.Layout(margin='5px')
    )
    
    # Button container
    button_container = widgets.HBox(
        [drive_button, directory_button],
        layout=widgets.Layout(
            display='flex',
            flex_flow='row',
            justify_content='flex-start',
            margin='10px 0'
        )
    )
    
    # Progress tracking
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=10,
        description='Proses:',
        layout=widgets.Layout(
            width='100%',
            margin='10px 0',
            visibility='hidden'  # Hidden by default
        ),
        style={'description_width': 'initial', 'bar_color': COLORS['primary']}
    )
    
    progress_message = widgets.HTML(
        value="",
        layout=widgets.Layout(
            margin='5px 0',
            visibility='hidden'  # Hidden by default
        )
    )
    
    progress_container = widgets.VBox(
        [progress_bar, progress_message],
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Status output area
    status = widgets.Output(
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
    
    # Reset progress helper
    def reset_progress():
        """Reset progress bar."""
        progress_bar.layout.visibility = 'hidden'
        progress_message.layout.visibility = 'hidden'
        progress_bar.value = 0
        progress_message.value = ""
    
    # Info box
    info_box = get_environment_info()
    
    # Container utama dengan semua komponen
    main = widgets.VBox(
        [
            header,
            colab_panel,
            button_container,
            progress_container,
            status,
            info_box
        ],
        layout=widgets.Layout(
            width='100%',
            padding='10px'
        )
    )
    
    # Struktur final komponen UI dengan struktur objek yang konsisten
    ui_components = {
        'ui': main,
        'colab_panel': colab_panel,
        'drive_button': drive_button,
        'directory_button': directory_button,
        'progress_bar': progress_bar,
        'progress_message': progress_message,
        'status': status,
        'progress_container': progress_container,
        'reset_progress': reset_progress,
        'module_name': 'env_config'
    }
    
    return ui_components