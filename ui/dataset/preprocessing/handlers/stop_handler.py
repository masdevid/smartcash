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
        # Set flag stop_requested di UI components
        ui_components['stop_requested'] = True
        
        # Set flag observer untuk memberi sinyal stop
        observer_manager = ui_components.get('observer_manager')
        if observer_manager and hasattr(observer_manager, 'set_flag'):
            observer_manager.set_flag('stop_requested', True)
        
        # Log stop preprocessing
        log_message(ui_components, "Menghentikan preprocessing dataset...", "warning", "⏹️")
        
        # Get notification manager
        from smartcash.ui.dataset.preprocessing.utils.notification_manager import get_notification_manager
        notification_manager = get_notification_manager(ui_components)
        
        # Notify process stop menggunakan notification manager
        notification_manager.notify_process_stop("Stop oleh pengguna")
        
        # Sembunyikan tombol stop
        if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
            ui_components['stop_button'].layout.display = 'none'
            
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
    
    # Set flag preprocessing_running ke False
    ui_components['preprocessing_running'] = False
    
    # Reset state preprocessing
    set_preprocessing_state(ui_components, False)
    
    # Update UI state menggunakan notification manager
    try:
        from smartcash.ui.dataset.preprocessing.utils.notification_manager import get_notification_manager
        notification_manager = get_notification_manager(ui_components)
        notification_manager.update_status("warning", "Preprocessing dihentikan oleh pengguna")
    except Exception:
        # Fallback ke update_ui_state jika notification manager tidak tersedia
        update_ui_state(ui_components, "warning", "Preprocessing dihentikan oleh pengguna")
    
    # Update confirmation area jika tersedia
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        # Bersihkan dan sembunyikan confirmation area
        ui_components['confirmation_area'].clear_output(wait=True)
        if hasattr(ui_components['confirmation_area'], 'layout'):
            ui_components['confirmation_area'].layout.display = 'none'
    
    # Reset UI setelah operasi
    reset_after_operation(ui_components)
    
    # Hilangkan flag stop request
    ui_components['stop_requested'] = False 