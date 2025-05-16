"""
File: smartcash/ui/setup/env_config_component.py
Deskripsi: Komponen UI untuk konfigurasi environment
"""

import ipywidgets as widgets
from typing import Dict, Any, Callable, Optional, List, Tuple

from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.alert_utils import create_info_alert
from smartcash.ui.info_boxes import get_environment_info
from smartcash.ui.components.accordion_factory import create_accordion

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
        "🔧 Konfigurasi Environment", 
        "Setup environment dan direktori untuk SmartCash"
    )
    
    # Buat info alert dengan informasi yang lebih jelas
    info_alert = create_info_alert(
        "<b>Konfigurasi Environment</b> - Langkah ini diperlukan untuk menjalankan aplikasi SmartCash dengan benar. "
        "Sistem akan melakukan pemeriksaan environment dan sinkronisasi konfigurasi secara otomatis."
    )
    
    # Buat info environment dengan tampilan yang lebih informatif
    env_info = get_environment_info(env_manager)
    
    # Buat tombol untuk connect ke Google Drive dengan style yang lebih compact
    drive_button = widgets.Button(
        description="Hubungkan Drive",
        button_style="primary",
        icon="cloud",
        tooltip="Hubungkan ke Google Drive untuk menyimpan dataset dan model",
        layout=widgets.Layout(width='auto', padding='4px 10px', margin='0px')
    )
    
    # Buat tombol untuk setup direktori dengan style yang lebih compact
    directory_button = widgets.Button(
        description="Setup Direktori",
        button_style="info",
        icon="folder-plus",
        tooltip="Buat struktur direktori yang diperlukan untuk aplikasi",
        layout=widgets.Layout(width='auto', padding='4px 10px', margin='0px')
    )
    
    # Tombol cek environment dan simpan konfigurasi dihapus karena akan dilakukan otomatis
    
    # Buat progress bar dengan style yang lebih konsisten
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=10,
        description="Progress:",
        style={"description_width": "initial", "bar_color": "#3498db"},
        layout=widgets.Layout(width="100%", margin="10px 0", visibility="hidden")
    )
    
    # Buat label untuk progress dengan style yang lebih informatif
    progress_message = widgets.HTML(
        value="",
        layout=widgets.Layout(width="100%", margin="5px 0", visibility="hidden")
    )
    
    # Buat output widget untuk status dengan style yang lebih baik
    status = widgets.Output(layout=widgets.Layout(width="100%", margin="10px 0", padding="5px", border="1px solid #eee"))
    
    # Buat container untuk tombol dengan layout yang lebih rapi
    button_container = widgets.HBox([
        drive_button, 
        directory_button
    ], layout=widgets.Layout(width="100%", justify_content="center", margin="15px 0", gap="20px"))
    
    # Buat container untuk progress dengan padding yang lebih baik
    progress_container = widgets.VBox([
        progress_bar,
        progress_message
    ], layout=widgets.Layout(width="100%", padding="5px 0"))
    
    # Buat accordion untuk informasi environment (tertutup secara default)
    env_info_accordion = create_accordion([
        ("Informasi Environment", env_info)
    ], selected_index=None)  # None berarti semua section tertutup
    
    # Buat container untuk bagian utama (tanpa footer)
    main_content = widgets.VBox([
        header,
        info_alert,
        button_container,
        progress_container,
        status
    ], layout=widgets.Layout(width="100%", padding="10px 0", spacing="10px"))
    
    # Buat footer dengan accordion info environment
    footer = widgets.Box(
        [env_info_accordion],
        layout=widgets.Layout(width="100%", margin="20px 0 0 0", padding="10px", border_top="1px solid #e0e0e0")
    )
    
    # Buat container utama dengan main content dan footer
    main_container = widgets.VBox([
        main_content,
        footer
    ], layout=widgets.Layout(width="100%", padding="10px 0", spacing="0px"))
    
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
