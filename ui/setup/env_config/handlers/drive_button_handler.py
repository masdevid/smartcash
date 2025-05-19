"""
File: smartcash/ui/setup/env_config/handlers/drive_button_handler.py
Deskripsi: Handler untuk tombol Google Drive
"""

import ipywidgets as widgets
from typing import Dict, Any

from smartcash.ui.setup.env_config.utils.environment_detector import detect_environment
from smartcash.ui.setup.env_config.utils.ui_helpers import disable_ui_during_processing, cleanup_ui
from smartcash.ui.utils.alert_utils import create_info_box
from smartcash.common.environment import get_environment_manager
from smartcash.ui.utils.ui_logger import log_to_ui

def setup_drive_button_handler(ui_components: Dict[str, Any]) -> None:
    """
    Setup handler untuk tombol connect drive
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    # Register handler
    if 'drive_button' in ui_components:
        ui_components['drive_button'].on_click(
            lambda b: on_drive_button_click(b, ui_components)
        )

def on_drive_button_click(b: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol connect drive
    
    Args:
        b: Button widget
        ui_components: Dictionary berisi komponen UI
    """
    # Simpan status tombol Drive
    drive_button_disabled = ui_components['drive_button'].disabled
    drive_button_description = ui_components['drive_button'].description
    drive_button_tooltip = ui_components['drive_button'].tooltip
    drive_button_icon = ui_components['drive_button'].icon
    
    # Nonaktifkan UI selama proses
    disable_ui_during_processing(ui_components, True)
    
    # Update status panel
    ui_components['status_panel'].value = create_info_box(
        "Menghubungkan Drive", 
        "Sedang menghubungkan ke Google Drive...",
        style="info"
    ).value
    
    # Log info
    log_to_ui(ui_components, "Menghubungkan ke Google Drive...", "info", "🔄")
    
    try:
        # Dapatkan environment manager
        env_manager = get_environment_manager()
        
        # Mount drive
        success, message = env_manager.mount_drive()
        
        if success:
            # Log success
            log_to_ui(ui_components, message, "success", "✅")
            
            # Nonaktifkan tombol
            ui_components['drive_button'].disabled = True
            ui_components['drive_button'].description = "Drive Terhubung"
            ui_components['drive_button'].tooltip = "Google Drive sudah terhubung"
            ui_components['drive_button'].icon = "check"
            
            # Update status panel
            ui_components['status_panel'].value = create_info_box(
                "Drive Terhubung", 
                "Google Drive berhasil terhubung.",
                style="success"
            ).value
            
            # Detect environment lagi
            detect_environment(ui_components, env_manager)
        else:
            # Log error
            log_to_ui(ui_components, message, "error", "❌")
            
            # Update status panel
            ui_components['status_panel'].value = create_info_box(
                "Error", 
                f"Gagal menghubungkan ke Google Drive: {message}",
                style="error"
            ).value
    except Exception as e:
        # Log error
        log_to_ui(ui_components, f"Error: {str(e)}", "error", "❌")
        
        # Update status panel
        ui_components['status_panel'].value = create_info_box(
            "Error", 
            f"Terjadi kesalahan: {str(e)}",
            style="error"
        ).value
    finally:
        # Cleanup UI dengan mempertahankan status tombol Drive
        cleanup_ui(ui_components)
        
        # Kembalikan status tombol Drive jika sebelumnya disabled
        if drive_button_disabled:
            ui_components['drive_button'].disabled = True
            ui_components['drive_button'].description = drive_button_description
            ui_components['drive_button'].tooltip = drive_button_tooltip
            ui_components['drive_button'].icon = drive_button_icon
