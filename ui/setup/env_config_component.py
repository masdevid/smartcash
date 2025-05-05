"""
File: smartcash/ui/setup/env_config_component.py
Deskripsi: Komponen UI untuk konfigurasi environment dengan pemanfaatan implementasi DRY
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_env_config_ui(env, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat komponen UI untuk konfigurasi environment dengan implementasi DRY.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI standar
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.utils.layout_utils import STANDARD_LAYOUTS, OUTPUT_WIDGET, BUTTON
    from smartcash.ui.info_boxes import get_environment_info
    
    # Header
    header = create_header(
        "⚙️ Konfigurasi Environment", 
        "Setup environment sistem untuk SmartCash"
    )
    
    # Panel Colab Info dengan style dari constants
    info_style = 'info'
    from smartcash.ui.utils.alert_utils import create_info_alert
    colab_panel = create_info_alert(
        "ℹ️ Mendeteksi environment...",
        info_style
    )
    
    # Tombol dengan layout standar
    drive_button = widgets.Button(
        description='Hubungkan Google Drive',
        button_style='info',
        icon='link',
        tooltip="Hubungkan Google Drive untuk penyimpanan dataset dan konfigurasi",
        layout=widgets.Layout(margin='5px', display='none')  # Hidden by default
    )
    
    directory_button = widgets.Button(
        description='Setup Direktori Lokal',
        button_style='primary',
        icon='folder-plus',
        tooltip="Buat struktur direktori untuk SmartCash",
        layout=BUTTON
    )
    
    # Container tombol menggunakan STANDARD_LAYOUTS
    button_container = widgets.HBox(
        [drive_button, directory_button],
        layout=STANDARD_LAYOUTS['hbox']
    )
    
    # Progress tracking menggunakan style dari constants
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
        layout=STANDARD_LAYOUTS['vbox']
    )
    
    # Status output area menggunakan layout standar
    status = widgets.Output(
        layout=OUTPUT_WIDGET
    )
    
    # Reset progress helper
    def reset_progress():
        """Reset progress bar."""
        progress_bar.layout.visibility = 'hidden'
        progress_message.layout.visibility = 'hidden'
        progress_bar.value = 0
        progress_message.value = ""
    
    # Info box menggunakan komponen yang ada
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
        layout=STANDARD_LAYOUTS['container']
    )
    
    # Struktur final komponen UI
    ui_components = {
        'ui': main,
        'colab_panel': colab_panel,
        'drive_button': drive_button,
        'directory_button': directory_button,
        'button_container': button_container,
        'progress_bar': progress_bar,
        'progress_message': progress_message,
        'status': status,
        'progress_container': progress_container,
        'reset_progress': reset_progress,
        'module_name': 'env_config'
    }
    
    return ui_components