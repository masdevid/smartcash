"""
File: smartcash/ui/setup/env_config_component.py
Deskripsi: Komponen UI untuk konfigurasi environment
"""

import ipywidgets as widgets
from typing import Dict, Any, Callable, Optional, List, Tuple

from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.alert_utils import create_info_box
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.info_boxes import get_environment_info
from smartcash.ui.components.accordion_factory import create_accordion
from smartcash.ui.helpers.ui_helpers import create_spacing

def create_env_config_ui(env_manager: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat komponen UI untuk konfigurasi environment
    
    Args:
        env_manager: Environment manager
        config: Konfigurasi aplikasi
    
    Returns:
        Dictionary berisi komponen UI
    """
    # Header dengan komponen standar seperti dependency_installer
    header = create_header(
        "📂 Konfigurasi Environment", 
        "Setup environment dan direktori untuk SmartCash"
    )
    
    # Buat info alert dengan informasi yang lebih jelas menggunakan komponen yang sama dengan dependency_installer
    status_panel = widgets.HTML(
        create_info_box(
            "Konfigurasi Environment", 
            "Sistem akan melakukan pemeriksaan environment dan sinkronisasi konfigurasi secara otomatis.",
            style="info"
        ).value
    )
    
    # Buat info environment dengan tampilan yang lebih informatif
    env_info = get_environment_info(env_manager)
    
    # Buat tombol untuk connect ke Google Drive dengan style yang konsisten dengan dependency_installer
    drive_button = widgets.Button(
        description="Hubungkan Drive",
        button_style="primary",
        icon="cloud",
        tooltip="Hubungkan ke Google Drive untuk menyimpan dataset dan model",
        layout=widgets.Layout(margin='5px')
    )
    
    # Buat tombol untuk setup direktori dengan style yang konsisten dengan dependency_installer
    directory_button = widgets.Button(
        description="Setup Direktori",
        button_style="info",
        icon="folder-plus",
        tooltip="Buat struktur direktori yang diperlukan untuk aplikasi",
        layout=widgets.Layout(margin='5px')
    )
    
    # Tombol cek environment dan simpan konfigurasi dihapus karena akan dilakukan otomatis
    
    # Buat progress bar dengan style yang konsisten dengan dependency_installer
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=10,
        description="Progress:",
        style={"description_width": "initial", "bar_color": COLORS['primary']},
        layout=widgets.Layout(width="100%", margin="10px 0", visibility="hidden")
    )
    
    # Buat label untuk progress dengan style yang konsisten dengan dependency_installer
    progress_message = widgets.HTML(
        value="",
        layout=widgets.Layout(margin='5px 0', visibility="hidden")
    )
    
    # Buat output widget untuk status dengan style yang konsisten dengan dependency_installer
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
    
    # Buat container untuk tombol dengan layout yang konsisten dengan dependency_installer
    button_container = widgets.HBox(
        [drive_button, directory_button],
        layout=widgets.Layout(
            display='flex',
            flex_flow='row',
            justify_content='center',
            width='100%',
            margin='10px 0'
        )
    )
    
    # Buat container untuk progress dengan layout yang konsisten dengan dependency_installer
    progress_container = widgets.VBox([
        progress_bar,
        progress_message
    ], layout=widgets.Layout(width="100%", margin="5px 0"))
    
    # Buat accordion untuk informasi environment (tertutup secara default)
    env_info_accordion = create_accordion([
        ("Informasi Environment", env_info)
    ], selected_index=None)  # None berarti semua section tertutup
    
    # Buat container utama dengan semua komponen
    main_container = widgets.VBox(
        [
            header,
            status_panel,
            button_container,
            progress_container,
            status,
            create_spacing(10),
            env_info_accordion
        ],
        layout=widgets.Layout(
            width='100%',
            padding='10px'
        )
    )
    
    # Fungsi untuk reset progress
    def reset_progress():
        progress_bar.value = 0
        progress_bar.layout.visibility = "hidden"
        progress_message.value = ""
        progress_message.layout.visibility = "hidden"
    
    # Buat dictionary untuk menyimpan komponen UI (tanpa check_button dan save_button)
    ui_components = {
        "ui": main_container,
        "drive_button": drive_button,
        "directory_button": directory_button,
        "progress_bar": progress_bar,
        "progress_message": progress_message,
        "status": status,
        "status_panel": status_panel,
        "reset_progress": reset_progress,
        "module_name": "env_config"
    }
    
    return ui_components
