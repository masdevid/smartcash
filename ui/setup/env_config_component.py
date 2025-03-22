"""
File: smartcash/ui/setup/env_config_component.py
Deskripsi: Komponen UI untuk konfigurasi environment SmartCash dengan perbaikan struktur progress
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_env_config_ui(env, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat komponen UI untuk environment config dengan styling standar dan visibilitas progress yang diperbaiki.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI dari ui_helpers
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.info_boxes.environment_info import get_environment_info
    
    # Header
    header = create_header(
        "⚙️ Konfigurasi Environment", 
        "Setup environment untuk project SmartCash"
    )
    
    # Progress tracker dengan layout yang ditingkatkan - PERBAIKAN: Gunakan named components
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=3,
        description='Progress:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%', visibility='visible')
    )
    
    progress_message = widgets.HTML(
        value="Siap digunakan",
        layout=widgets.Layout(margin='0 0 0 10px', padding='5px')
    )
    
    progress_container = widgets.HBox(
        [progress_bar, progress_message], 
        layout=widgets.Layout(margin='10px 0', align_items='center')
    )
    
    # Panel status Colab/Drive
    colab_panel = widgets.HTML(value="Mendeteksi environment...")
    
    # Panel log messages dengan styling yang dioptimalkan untuk visibilitas
    log_header = widgets.HTML(f"<h3 style='margin: 10px 0 5px 0; color: {COLORS['dark']};'>{ICONS['file']} Log Messages</h3>")
    
    status = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border=f'2px solid {COLORS["primary"]}',  # Border lebih terlihat
            min_height='150px',          # Tinggi minimum lebih besar
            max_height='300px',
            margin='0 0 10px 0',
            padding='10px',
            overflow='auto',
            background_color=f'{COLORS["light"]}'   # Latar belakang untuk visibilitas
        )
    )
    
    # Wrapper untuk log agar lebih terlihat
    log_box = widgets.VBox([
        log_header,
        status
    ], layout=widgets.Layout(margin='15px 0'))
    
    # Panel info bantuan dengan menggunakan komponen standar
    help_box = get_environment_info()
    
    # Gunakan create_button_group dari ui_helpers untuk konsistensi
    try:
        from smartcash.ui.helpers.ui_helpers import create_button_group
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
        drive_button = button_group.children[0]
        directory_button = button_group.children[1]
    except ImportError:
        # Fallback manual jika ui_helpers tidak tersedia
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
        
        button_group = widgets.HBox([drive_button, directory_button],
            layout=widgets.Layout(
                display='flex',
                flex_flow='row wrap',
                gap='10px',
                margin='10px 0'
            )
        )
    
    # Gunakan divider dari ui_helpers
    try:
        from smartcash.ui.helpers.ui_helpers import create_divider
        divider = create_divider()
    except ImportError:
        # Fallback manual jika ui_helpers tidak tersedia
        divider = widgets.HTML(f"<hr style='margin: 15px 0; border: 0; border-top: 1px solid {COLORS['border']};'>")
    
    # Container utama - dengan urutan diubah untuk meningkatkan visibilitas log
    ui = widgets.VBox([
        header,
        colab_panel,
        progress_container,
        divider, 
        button_group,
        log_box,              # Taruh log box sebelum help_box
        help_box              # Help box di paling bawah
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Komponen UI dengan referensi ke status untuk logging
    ui_components = {
        'ui': ui,
        'header': header,
        'colab_panel': colab_panel,
        'drive_button': drive_button,
        'directory_button': directory_button,
        'progress_bar': progress_bar,  # PERBAIKAN: Referensi langsung ke progress_bar
        'progress_message': progress_message,  # PERBAIKAN: Referensi langsung ke progress_message
        'progress_container': progress_container,
        'status': status,
        'log_header': log_header,
        'log_box': log_box,
        'help_panel': help_box,
        'module_name': 'env_config'  # Tambahkan module_name untuk memudahkan setup logger
    }
    
    # Test log langsung ke widget untuk memastikan fungsi
    from IPython.display import display, HTML
    with status:
        display(HTML(f"<div style='color: {COLORS['primary']};'><strong>🚀 UI komponen environment config berhasil dibuat</strong></div>"))
    
    return ui_components