"""
File: smartcash/ui/handlers/processing_cleanup_handler.py
Deskripsi: Handler pembersihan untuk modul preprocessing dan augmentasi
"""

from typing import Dict, Any, Optional, Callable, Union, List
from IPython.display import display, clear_output
import os
import traceback
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.common.logger import get_logger

def _update_status_panel(ui_components: Dict[str, Any], status_type: str, message: str) -> None:
    """
    Update panel status dengan pesan dan tipe tertentu.
    
    Args:
        ui_components: Dictionary komponen UI
        status_type: Tipe status ('info', 'success', 'warning', 'error')
        message: Pesan yang akan ditampilkan
    """
    if 'status' in ui_components:
        with ui_components['status']:
            # Tidak menggunakan clear_output agar output sebelumnya tetap terlihat
            display(create_status_indicator(status_type, message))

def cleanup_ui(ui_components: Dict[str, Any], module_type: str = 'preprocessing') -> None:
    """
    Membersihkan UI setelah proses selesai.
    
    Args:
        ui_components: Dictionary komponen UI
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
    """
    logger = ui_components.get('logger', get_logger(module_type))
    
    try:
        # Reset progress bar
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].description = ''
        
        # Reset label progress
        for label_key in ['current_progress', 'overall_label', 'step_label']:
            if label_key in ui_components and hasattr(ui_components[label_key], 'value'):
                ui_components[label_key].value = ''
        
        # Enable tombol yang dinonaktifkan
        for btn_key in ['primary_button', 'reset_button', 'save_button', 'cleanup_button']:
            if btn_key in ui_components and hasattr(ui_components[btn_key], 'disabled'):
                ui_components[btn_key].disabled = False
        
        # Set flag running ke False
        running_flag_key = f"{module_type}_running"
        ui_components[running_flag_key] = False
        
        logger.debug(f"✅ UI berhasil dibersihkan setelah proses {module_type}")
    except Exception as e:
        logger.error(f"❌ Error saat membersihkan UI: {str(e)}")
        logger.error(traceback.format_exc())

def setup_processing_cleanup_handler(
    ui_components: Dict[str, Any], 
    module_type: str = 'preprocessing',
    config: Dict[str, Any] = None, 
    env = None
) -> Dict[str, Any]:
    """
    Setup handler untuk pembersihan UI dan proses.
    
    Args:
        ui_components: Dictionary komponen UI
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        config: Konfigurasi aplikasi
        env: Environment manager
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    logger = ui_components.get('logger', get_logger(module_type))
    
    try:
        # Tambahkan fungsi cleanup ke ui_components
        ui_components['cleanup_ui'] = lambda: cleanup_ui(ui_components, module_type)
        
        # Tambahkan fungsi update status panel
        ui_components['update_status'] = lambda status_type, message: _update_status_panel(ui_components, status_type, message)
        
        # Set flag running ke False
        running_flag_key = f"{module_type}_running"
        ui_components[running_flag_key] = False
        
        logger.debug(f"✅ Handler cleanup untuk {module_type} berhasil disetup")
    except Exception as e:
        logger.error(f"❌ Error saat setup handler cleanup: {str(e)}")
        logger.error(traceback.format_exc())
    
    return ui_components
