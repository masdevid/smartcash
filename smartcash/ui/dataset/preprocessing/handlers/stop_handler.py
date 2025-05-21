"""
File: smartcash/ui/dataset/preprocessing/handlers/stop_handler.py
Deskripsi: Handler untuk tombol stop preprocessing dataset
"""

from typing import Dict, Any, Optional

from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import (
    update_ui_state, update_status_panel, set_preprocessing_state, reset_after_operation
)
from smartcash.ui.dataset.preprocessing.utils.ui_observers import notify_process_stop

def handle_stop_button_click(button: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol stop preprocessing.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    # Disable tombol untuk mencegah multiple click
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        # Set flag observer untuk memberi sinyal stop
        observer_manager = ui_components.get('observer_manager')
        if observer_manager and hasattr(observer_manager, 'set_flag'):
            observer_manager.set_flag('stop_requested', True)
        
        # Log stop preprocessing
        log_message(ui_components, "Menghentikan preprocessing dataset...", "warning", "⏹️")
        
        # Update UI state
        update_status_panel(ui_components, "warning", "Menghentikan preprocessing...")
        
        # Tambah flag stop request
        ui_components['stop_requested'] = True
        
        # Notify process stop
        notify_process_stop(ui_components, "Stop oleh pengguna")
        
        # Reset state preprocessing
        stop_preprocessing(ui_components)
        
    except Exception as e:
        # Log error
        error_message = str(e)
        update_ui_state(ui_components, "error", f"Error saat menghentikan preprocessing: {error_message}")
        log_message(ui_components, f"Error saat menghentikan preprocessing: {error_message}", "error", "❌")
    
    finally:
        # Re-enable tombol setelah operasi selesai
        if button and hasattr(button, 'disabled'):
            button.disabled = False

def stop_preprocessing(ui_components: Dict[str, Any]) -> None:
    """
    Menghentikan proses preprocessing yang sedang berjalan.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Log menghentikan preprocessing
    log_message(ui_components, "Preprocessing dihentikan oleh pengguna", "warning", "⏹️")
    
    # Reset state preprocessing
    set_preprocessing_state(ui_components, False)
    
    # Update UI state
    update_ui_state(ui_components, "warning", "Preprocessing dihentikan oleh pengguna")
    
    # Reset UI setelah operasi
    reset_after_operation(ui_components)
    
    # Hilangkan flag stop request
    ui_components['stop_requested'] = False 