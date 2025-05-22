"""
File: smartcash/ui/dataset/augmentation/handlers/stop_handler.py
Deskripsi: Handler untuk menghentikan proses augmentasi
"""

from typing import Dict, Any, Optional
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.augmentation.utils.ui_state_manager import update_status_panel, disable_buttons
from smartcash.ui.dataset.augmentation.utils.ui_observers import notify_process_stop

def handle_stop_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler untuk tombol stop augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget (opsional)
    """
    try:
        # Setup logger jika belum
        ui_components = setup_ui_logger(ui_components)
        
        # Set flag untuk menghentikan proses
        ui_components['augmentation_running'] = False
        ui_components['stop_requested'] = True
        
        # Update status
        update_status_panel(ui_components, "Menghentikan augmentasi...", "warning")
        log_message(ui_components, "Menghentikan proses augmentasi...", "warning", "⏹️")
        
        # Notifikasi observer
        notify_process_stop(ui_components)
        
        # Sembunyikan tombol stop, tampilkan tombol augment
        if 'stop_button' in ui_components:
            ui_components['stop_button'].layout.display = 'none'
        if 'augment_button' in ui_components:
            ui_components['augment_button'].layout.display = 'block'
            ui_components['augment_button'].disabled = False
        
        # Enable tombol lainnya
        disable_buttons(ui_components, False)
        
        # Update status final
        update_status_panel(ui_components, "Augmentasi dihentikan oleh pengguna", "info")
        log_message(ui_components, "Augmentasi berhasil dihentikan", "info", "✅")
        
    except Exception as e:
        log_message(ui_components, f"Error saat menghentikan augmentasi: {str(e)}", "error", "❌")
        update_status_panel(ui_components, f"Error saat menghentikan: {str(e)}", "error")

def stop_augmentation(ui_components: Dict[str, Any]) -> None:
    """
    Hentikan proses augmentasi yang sedang berjalan.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Setup logger jika belum
    ui_components = setup_ui_logger(ui_components)
    
    # Set flag stop
    ui_components['stop_requested'] = True
    ui_components['augmentation_running'] = False
    
    log_message(ui_components, "Permintaan stop augmentasi diterima", "warning", "⏹️")
    
    # Notifikasi service jika ada
    if 'augmentation_service' in ui_components and ui_components['augmentation_service']:
        try:
            service = ui_components['augmentation_service']
            if hasattr(service, 'stop'):
                service.stop()
            elif hasattr(service, 'cancel'):
                service.cancel()
        except Exception as e:
            log_message(ui_components, f"Gagal menghentikan service: {str(e)}", "warning", "⚠️")
    
    # Reset UI
    from smartcash.ui.dataset.augmentation.utils.ui_state_manager import reset_ui_after_augmentation
    reset_ui_after_augmentation(ui_components)

def is_stop_requested(ui_components: Dict[str, Any]) -> bool:
    """
    Cek apakah ada permintaan untuk menghentikan proses.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        bool: True jika ada permintaan stop
    """
    return ui_components.get('stop_requested', False)

def clear_stop_request(ui_components: Dict[str, Any]) -> None:
    """
    Bersihkan flag stop request.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    ui_components['stop_requested'] = False