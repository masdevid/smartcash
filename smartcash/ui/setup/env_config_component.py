"""
File: smartcash/ui/setup/env_config_component.py
Deskripsi: Komponen UI untuk konfigurasi environment
"""

import ipywidgets as widgets
from typing import Dict, Any, Callable, Optional

from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.alert_utils import create_info_alert
from smartcash.ui.info_boxes import get_environment_info

def create_env_config_ui(env_manager: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat komponen UI untuk konfigurasi environment
    
    Args:
        env_manager: Environment manager
        config: Konfigurasi aplikasi
    
    Returns:
        Dictionary berisi komponen UI
    """
    # Buat header
    header = create_header("Konfigurasi Environment", "🔧", "Konfigurasi environment untuk SmartCash")
    
    # Buat info alert
    info_alert = create_info_alert(
        "Konfigurasi ini diperlukan untuk menjalankan aplikasi SmartCash. "
        "Pastikan semua persyaratan terpenuhi sebelum melanjutkan."
    )
    
    # Buat info environment
    env_info = get_environment_info(env_manager)
    
    # Buat tombol untuk connect ke Google Drive
    drive_button = widgets.Button(
        description="Connect Google Drive",
        button_style="primary",
        icon="cloud",
        tooltip="Hubungkan ke Google Drive untuk menyimpan dataset dan model"
    )
    
    # Buat tombol untuk setup direktori
    directory_button = widgets.Button(
        description="Setup Direktori",
        button_style="info",
        icon="folder-plus",
        tooltip="Buat struktur direktori yang diperlukan"
    )
    
    # Tombol cek environment dan simpan konfigurasi dihapus karena akan dilakukan otomatis
    
    # Buat progress bar
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=10,
        description="Progress:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="100%", visibility="hidden")
    )
    
    # Buat label untuk progress
    progress_message = widgets.HTML(
        value="",
        layout=widgets.Layout(width="100%", visibility="hidden")
    )
    
    # Buat output widget untuk status
    status = widgets.Output()
    
    # Buat container untuk tombol (hanya drive dan direktori)
    button_container = widgets.HBox([
        drive_button, 
        directory_button
    ], layout=widgets.Layout(width="100%", justify_content="space-around"))
    
    # Buat container untuk progress
    progress_container = widgets.VBox([
        progress_bar,
        progress_message
    ], layout=widgets.Layout(width="100%"))
    
    # Buat container utama
    main_container = widgets.VBox([
        header,
        info_alert,
        env_info,
        button_container,
        progress_container,
        status
    ], layout=widgets.Layout(width="100%"))
    
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
        "reset_progress": reset_progress,
        "module_name": "env_config"
    }
    
    return ui_components
