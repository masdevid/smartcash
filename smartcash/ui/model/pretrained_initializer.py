"""
File: smartcash/ui/model/pretrained_initializer.py
Deskripsi: Inisialisasi UI dan logika bisnis untuk pretrained model dengan pendekatan DRY yang disederhanakan
"""

from typing import Dict, Any, Optional, Callable
from IPython.display import display, clear_output
import os
from pathlib import Path
import time
import threading
import shutil

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def initialize_pretrained_model_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI untuk pretrained model dengan tombol Download & Sync.
    
    Returns:
        Dictionary berisi komponen UI yang telah diinisialisasi
    """
    try:
        # Import komponen UI
        from smartcash.ui.model.components.pretrained_components import create_pretrained_ui
        
        # Buat komponen UI
        ui_components = create_pretrained_ui()
        
        # Tampilkan UI
        clear_output(wait=True)
        display(ui_components['main_container'])
        
        # Setup handler untuk tombol download & sync
        from smartcash.ui.model.handlers.simple_download import handle_download_sync_button
        
        # Register handler untuk tombol download & sync
        if 'download_sync_button' in ui_components and ui_components['download_sync_button']:
            ui_components['download_sync_button'].on_click(
                lambda b: handle_download_sync_button(b, ui_components)
            )
            logger.debug(f"{ICONS.get('link', '🔗')} Handler untuk tombol download & sync terdaftar")
        
        return ui_components
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat inisialisasi UI pretrained model: {str(e)}")
        
        # Import widgets jika belum diimpor
        import ipywidgets as widgets
        
        # Buat container minimal untuk menampilkan error
        error_container = widgets.VBox([
            widgets.HTML(f"<h3>{ICONS.get('error', '❌')} Error saat inisialisasi UI pretrained model</h3>"),
            widgets.HTML(f"<p>{str(e)}</p>")
        ])
        
        display(error_container)
        
        return {'main_container': error_container}

# Fungsi untuk memeriksa apakah Google Drive terpasang
def is_drive_mounted() -> bool:
    """
    Memeriksa apakah Google Drive terpasang.
    
    Returns:
        True jika Google Drive terpasang, False jika tidak
    """
    return os.path.exists('/content/drive/MyDrive')

# Fungsi untuk memasang Google Drive
def mount_drive() -> tuple:
    """
    Memasang Google Drive.
    
    Returns:
        Tuple (success, message)
    """
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        return True, f"{ICONS.get('success', '✅')} Google Drive berhasil dipasang"
    except Exception as e:
        return False, f"{ICONS.get('error', '❌')} Gagal memasang Google Drive: {str(e)}"

# Fungsi utama untuk dijalankan di notebook
def setup_pretrained_model_ui():
    """
    Fungsi utama untuk setup UI pretrained model.
    Fungsi ini yang akan dipanggil dari notebook.
    """
    # Inisialisasi UI dan dapatkan komponen UI
    ui_components = initialize_pretrained_model_ui()
    
    # Kembalikan komponen UI
    return ui_components

# Fungsi untuk menjalankan test case
def run_tests():
    """
    Menjalankan test case untuk memastikan fungsi download dan sync berjalan dengan benar.
    """
    from smartcash.ui.model.tests.test_simple_download import run_all_tests
    return run_all_tests()
