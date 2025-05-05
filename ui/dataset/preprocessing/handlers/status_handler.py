"""
File: smartcash/ui/dataset/preprocessing/handlers/status_handler.py
Deskripsi: Handler untuk mengelola status panel pada UI preprocessing
"""

import ipywidgets as widgets
from typing import Dict, Any

# Import status handler utama untuk menjaga konsistensi
from smartcash.ui.handlers.status_handler import update_status_panel as global_update_status_panel
from smartcash.ui.handlers.status_handler import create_status_panel as global_create_status_panel
from smartcash.ui.utils.constants import ALERT_STYLES, ICONS, COLORS

def update_status_panel(ui_components: Dict[str, Any], status_type: str, message: str) -> None:
    """
    Update status panel UI dengan pesan dan tipe yang ditentukan.
    
    Args:
        ui_components: Dictionary komponen UI dengan kunci 'status_panel'
        status_type: Tipe pesan ('info', 'success', 'warning', 'error')
        message: Pesan yang akan ditampilkan
    """
    # Delegasikan ke fungsi global untuk konsistensi
    global_update_status_panel(ui_components, status_type, message)

def create_status_panel(message: str = "", status_type: str = "info") -> widgets.HTML:
    """
    Buat komponen status panel untuk modul preprocessing dengan styling yang konsisten.
    
    Args:
        message: Pesan awal untuk status
        status_type: Tipe status awal ('info', 'success', 'warning', 'error')
        
    Returns:
        Widget HTML berisi status panel
    """
    # Delegasikan ke fungsi global untuk konsistensi
    return global_create_status_panel(message, status_type)
