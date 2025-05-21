"""
File: smartcash/ui/dataset/preprocessing/utils/ui_state_manager.py
Deskripsi: Utilitas untuk mengelola state UI pada proses preprocessing dataset
"""

from typing import Dict, Any, Optional, List
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.dataset.preprocessing.utils.notification_manager import notify_status

def update_status_panel(ui_components: Dict[str, Any], status: str, message: str = "") -> None:
    """
    Update status panel dengan status dan pesan.
    
    Args:
        ui_components: Dictionary komponen UI
        status: Status baru (started, completed, failed, paused, resumed)
        message: Pesan status tambahan
    """
    # Skip jika ui_components tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Pastikan status panel tersedia
    if 'status_panel' not in ui_components:
        log_message(ui_components, "Status panel tidak tersedia", "debug", "ℹ️")
        return
    
    # Map status ke emoji dan warna
    status_map = {
        "started": ("🔄", "blue", "Running", "info"),
        "running": ("🔄", "blue", "Running", "info"),
        "completed": ("✅", "green", "Completed", "success"),
        "success": ("✅", "green", "Sukses", "success"),
        "failed": ("❌", "red", "Failed", "error"),
        "error": ("❌", "red", "Error", "error"),
        "paused": ("⏸️", "orange", "Paused", "warning"),
        "warning": ("⚠️", "orange", "Warning", "warning"),
        "reset": ("🔄", "gray", "Reset", "info"),
        "idle": ("ℹ️", "gray", "Idle", "info"),
        "loading": ("⏳", "blue", "Loading", "info")
    }
    
    # Get emoji, warna, dan default message berdasarkan status
    emoji, color, default_text, log_level = status_map.get(status.lower(), ("ℹ️", "gray", "Unknown", "info"))
    
    # Gunakan default message jika tidak ada pesan
    display_message = message or default_text
    
    # Update status panel dengan HTML
    ui_components['status_panel'].value = f"<span style='color: {color};'>{emoji} {display_message}</span>"
    
    # Log status update
    log_message(ui_components, f"Status: {display_message}", log_level, emoji)
    
    # Notifikasi status melalui observer
    notify_status(ui_components, status, message)

def reset_ui_after_preprocessing(ui_components: Dict[str, Any]) -> None:
    """
    Reset UI components setelah preprocessing selesai.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Skip jika ui_components tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Reset button states
    if 'preprocess_button' in ui_components and hasattr(ui_components['preprocess_button'], 'disabled'):
        ui_components['preprocess_button'].disabled = False
    
    if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'disabled'):
        ui_components['stop_button'].disabled = True
    
    if 'reset_button' in ui_components and hasattr(ui_components['reset_button'], 'disabled'):
        ui_components['reset_button'].disabled = False
    
    if 'cleanup_button' in ui_components and hasattr(ui_components['cleanup_button'], 'disabled'):
        ui_components['cleanup_button'].disabled = False
    
    if 'save_button' in ui_components and hasattr(ui_components['save_button'], 'disabled'):
        ui_components['save_button'].disabled = False
    
    # Reset flags
    ui_components['preprocessing_running'] = False
    
    # Log reset berhasil
    log_message(ui_components, "UI reset setelah preprocessing", "debug", "🔄")

def update_ui_before_preprocessing(ui_components: Dict[str, Any]) -> None:
    """
    Update UI components sebelum preprocessing dimulai.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Skip jika ui_components tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Update button states
    if 'preprocess_button' in ui_components and hasattr(ui_components['preprocess_button'], 'disabled'):
        ui_components['preprocess_button'].disabled = True
    
    if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'disabled'):
        ui_components['stop_button'].disabled = False
    
    if 'reset_button' in ui_components and hasattr(ui_components['reset_button'], 'disabled'):
        ui_components['reset_button'].disabled = True
    
    if 'cleanup_button' in ui_components and hasattr(ui_components['cleanup_button'], 'disabled'):
        ui_components['cleanup_button'].disabled = True
    
    if 'save_button' in ui_components and hasattr(ui_components['save_button'], 'disabled'):
        ui_components['save_button'].disabled = True
    
    # Update flags
    ui_components['preprocessing_running'] = True
    
    # Update status
    update_status_panel(ui_components, "started", "Preprocessing sedang berjalan...")
    
    # Log update berhasil
    log_message(ui_components, "UI dipersiapkan untuk preprocessing", "debug", "🔄")

def is_preprocessing_running(ui_components: Dict[str, Any]) -> bool:
    """
    Cek apakah preprocessing sedang berjalan.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        bool: True jika preprocessing sedang berjalan, False jika tidak
    """
    return ui_components.get('preprocessing_running', False)

def set_preprocessing_state(ui_components: Dict[str, Any], running: bool) -> None:
    """
    Set state preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        running: State running baru
    """
    # Skip jika ui_components tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Update flag
    ui_components['preprocessing_running'] = running
    
    # Update UI berdasarkan state
    if running:
        update_ui_before_preprocessing(ui_components)
    else:
        reset_ui_after_preprocessing(ui_components)

def toggle_input_controls(ui_components: Dict[str, Any], disabled: bool) -> None:
    """
    Toggle control input berdasarkan state.
    
    Args:
        ui_components: Dictionary komponen UI
        disabled: True untuk disable controls, False untuk enable
    """
    # Skip jika ui_components tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Identifikasi semua input controls
    input_controls = []
    
    # Tambahkan semua widget yang memiliki atribut disabled
    for key, widget in ui_components.items():
        if hasattr(widget, 'disabled') and key not in [
            'preprocess_button', 'stop_button', 'reset_button', 'cleanup_button', 'save_button'
        ]:
            input_controls.append(widget)
    
    # Toggle disabled state untuk semua controls
    for control in input_controls:
        control.disabled = disabled 