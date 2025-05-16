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
from smartcash.ui.utils.info_utils import create_info_accordion
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
    # Gunakan fungsi get_environment_info dengan parameter yang benar
    env_info_content = get_environment_info(open_by_default=False)
    
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
    
    # Progress bar dihilangkan sesuai permintaan
    
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
    
    # Buat container utama dengan semua komponen
    main_container = widgets.VBox(
        [
            header,
            status_panel,
            button_container,
            status,
            create_spacing(10),
            env_info_content
        ],
        layout=widgets.Layout(
            width='100%',
            padding='10px'
        )
    )
    
    # Fungsi reset progress tidak diperlukan lagi karena progress bar dihilangkan
    def reset_progress():
        pass  # Tetap disediakan untuk kompatibilitas
    
    # Buat dictionary untuk menyimpan komponen UI (tanpa check_button dan save_button)
    ui_components = {
        "ui": main_container,
        "drive_button": drive_button,
        "directory_button": directory_button,
        "status": status,
        "status_panel": status_panel,
        "reset_progress": reset_progress,
        "module_name": "env_config"
    }
    
    return ui_components
