"""
File: smartcash/ui/dataset/preprocessing/handlers/status_handler.py
Deskripsi: Handler untuk mengelola status panel pada UI preprocessing
"""

import ipywidgets as widgets
from typing import Dict, Any

# Import utils dari preprocessing module
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import update_status_panel
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.dataset.preprocessing.utils.notification_manager import notify_status
from smartcash.ui.utils.constants import ALERT_STYLES, ICONS, COLORS

def create_status_panel(message: str = "", status_type: str = "info") -> widgets.HTML:
    """
    Buat komponen status panel untuk modul preprocessing dengan styling yang konsisten.
    
    Args:
        message: Pesan awal untuk status
        status_type: Tipe status awal ('info', 'success', 'warning', 'error')
        
    Returns:
        Widget HTML berisi status panel
    """
    # Pemetaan status ke icon dan warna
    status_map = {
        "info": ("ℹ️", "blue", "Info"),
        "success": ("✅", "green", "Success"),
        "warning": ("⚠️", "orange", "Warning"),
        "error": ("❌", "red", "Error"),
        "loading": ("⏳", "blue", "Loading"),
        "idle": ("⏸️", "gray", "Idle"),
        "started": ("🚀", "blue", "Started"),
        "completed": ("✅", "green", "Completed"),
        "failed": ("❌", "red", "Failed")
    }
    
    emoji, color, default_text = status_map.get(status_type, ("ℹ️", "gray", "Info"))
    
    # Gunakan default message jika tidak ada pesan
    display_message = message or default_text
    
    # Buat HTML widget dengan pesan dan style yang sesuai
    html = widgets.HTML(
        value=f"<span style='color: {color};'>{emoji} {display_message}</span>"
    )
    
    return html

def setup_status_handler(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk status panel preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Pastikan status panel tersedia
    if 'status_panel' not in ui_components:
        ui_components['status_panel'] = create_status_panel("Siap untuk preprocessing dataset", "idle")
    
    # Tambahkan fungsi ke ui_components jika belum ada
    ui_components['update_status_panel'] = lambda status, message: update_status_panel(ui_components, status, message)
    
    # Tambahkan fungsi create panel
    ui_components['create_status_panel'] = create_status_panel
    
    # Set status awal
    update_status_panel(ui_components, "idle", "Preprocessing siap dijalankan")
    
    # Notifikasi status awal
    notify_status(ui_components, "idle", "Preprocessing siap dijalankan")
    
    # Log setup berhasil
    log_message(ui_components, "Status handler berhasil disetup", "debug", "✅")
    
    return ui_components
