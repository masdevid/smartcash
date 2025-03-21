"""
File: smartcash/ui/setup/env_config_component.py
Deskripsi: Komponen UI untuk konfigurasi environment SmartCash dengan urutan komponen yang dioptimalkan
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_env_config_ui(env, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat komponen UI untuk environment config dengan styling standar dari ui_helpers dan progress tracker.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI dari ui_helpers
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.helpers.ui_helpers import create_button_group, create_divider
    from smartcash.ui.info_boxes.environment_info import get_environment_info
    
    # Header
    header = create_header(
        "⚙️ Konfigurasi Environment", 
        "Setup environment untuk project SmartCash"
    )
    
    # Progress tracker components
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=3,
        description='Progress:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%', margin='10px 0')
    )
    
    progress_message = widgets.HTML("Mempersiapkan environment...")
    
    # Panel status Colab/Drive
    colab_panel = widgets.HTML(value="Mendeteksi environment...")
    
    # Panel info bantuan dengan menggunakan komponen standar
    help_box = get_environment_info()
    
    # Gunakan create_button_group dari ui_helpers
    buttons = [
        ("Hubungkan Google Drive", "primary", "link", None),  # callback akan ditambahkan di handler
        ("Setup Direktori Lokal", "info", "folder-plus", None)
    ]
    
    button_group = create_button_group(buttons, 
        widgets.Layout(
            display='flex',
            flex_flow='row wrap',
            gap='10px',
            margin='10px 0'
        )
    )
    
    # Panel output
    status = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            padding='10px',
            overflow='auto'
        )
    )
    
    # Gunakan divider dari ui_helpers
    divider = create_divider()
    
    # Container utama - dengan urutan diubah sesuai permintaan:
    # 1. `help_panel` (help_box) diletakkan di paling bawah
    # 2. `progress_bar` dan `progress_message` diletakkan di bawah `colab_panel`
    ui = widgets.VBox([
        header,
        colab_panel,
        widgets.VBox([progress_bar, progress_message], layout=widgets.Layout(margin='10px 0')),
        divider, 
        button_group, 
        status,
        help_box  # Posisikan help_box di paling bawah
    ])
    
    # Komponen UI
    ui_components = {
        'ui': ui,
        'header': header,
        'colab_panel': colab_panel,
        'drive_button': button_group.children[0],
        'directory_button': button_group.children[1],
        'progress_bar': progress_bar,
        'progress_message': progress_message,
        'status': status,
        'help_panel': help_box
    }
    
    return ui_components