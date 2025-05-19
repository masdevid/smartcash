"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Komponen UI untuk konfigurasi environment
"""

import ipywidgets as widgets
from typing import Dict, Any
from datetime import datetime

from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.alert_utils import create_info_box
from smartcash.ui.utils.constants import COLORS
from smartcash.ui.info_boxes import get_environment_info
from smartcash.ui.helpers.ui_helpers import create_spacing

def create_env_config_ui(env_manager: Any, config_manager: Any) -> Dict[str, Any]:
    """
    Buat komponen UI untuk konfigurasi environment
    
    Args:
        env_manager: Environment manager
        config_manager: Konfigurasi manager
    
    Returns:
        Dictionary berisi komponen UI
    """
    # Header dengan komponen standar
    header = create_header(
        "📂 Konfigurasi Environment", 
        "Setup environment dan direktori untuk SmartCash"
    )
    
    # Panel status dengan informasi sinkronisasi
    last_sync = config_manager.get_last_sync_time()
    sync_status = "Terakhir sinkronisasi: " + (last_sync.strftime("%Y-%m-%d %H:%M:%S") if last_sync else "Belum pernah")
    
    status_panel = widgets.HTML(
        create_info_box(
            "Konfigurasi Environment", 
            f"Sistem akan melakukan pemeriksaan environment dan sinkronisasi konfigurasi secara otomatis.<br>{sync_status}",
            style="info"
        ).value
    )
    
    # Informasi environment
    env_info_content = get_environment_info(open_by_default=False)
    
    # Tombol untuk menghubungkan ke Google Drive
    drive_button = widgets.Button(
        description="Hubungkan Drive",
        button_style="primary",
        icon="cloud",
        tooltip="Hubungkan ke Google Drive untuk menyimpan dataset dan model",
        layout=widgets.Layout(margin='5px')
    )
    
    # Tombol untuk setup direktori
    directory_button = widgets.Button(
        description="Setup Direktori",
        button_style="info",
        icon="folder-plus",
        tooltip="Buat struktur direktori yang diperlukan untuk aplikasi",
        layout=widgets.Layout(margin='5px')
    )
    
    # Tombol untuk sinkronisasi manual
    sync_button = widgets.Button(
        description="Sinkronisasi",
        button_style="success",
        icon="sync",
        tooltip="Sinkronkan konfigurasi dengan Google Drive dan Colab",
        layout=widgets.Layout(margin='5px')
    )
    
    # Output widget untuk status
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
    
    # Progress bar untuk proses yang berjalan
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Progress:',
        bar_style='info',
        style={'description_width': 'initial'},
        layout=widgets.Layout(visibility='hidden', width='100%', margin='10px 0')
    )
    
    # Label untuk progress bar
    progress_message = widgets.HTML(
        value="",
        layout=widgets.Layout(visibility='hidden', margin='5px 0')
    )
    
    # Container untuk tombol
    button_container = widgets.HBox(
        [drive_button, directory_button, sync_button],
        layout=widgets.Layout(
            display='flex',
            flex_flow='row',
            justify_content='center',
            width='100%',
            margin='10px 0'
        )
    )
    
    # Container utama
    main_container = widgets.VBox(
        [
            header,
            status_panel,
            button_container,
            status,
            progress_bar,
            progress_message,
            create_spacing(10),
            env_info_content
        ],
        layout=widgets.Layout(
            width='100%',
            padding='10px'
        )
    )
    
    # Fungsi reset progress
    def reset_progress():
        progress_bar.value = 0
        progress_bar.layout.visibility = 'hidden'
        progress_message.value = ""
        progress_message.layout.visibility = 'hidden'
    
    # Dictionary komponen UI
    ui_components = {
        "ui": main_container,
        "drive_button": drive_button,
        "directory_button": directory_button,
        "sync_button": sync_button,
        "status": status,
        "status_panel": status_panel,
        "progress_bar": progress_bar,
        "progress_message": progress_message,
        "reset_progress": reset_progress,
        "module_name": "env_config"
    }
    
    return ui_components
