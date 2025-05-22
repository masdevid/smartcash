"""
File: smartcash/ui/dataset/augmentation/handlers/stop_handler.py
Deskripsi: Handler untuk menghentikan proses augmentasi dengan logger bridge (SRP)
"""

from typing import Dict, Any
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.dataset.augmentation.handlers.state_handler import StateHandler

def handle_stop_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler untuk tombol stop augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget (opsional)
    """
    ui_logger = create_ui_logger_bridge(ui_components, "stop_handler")
    state_handler = StateHandler(ui_components, ui_logger)
    
    try:
        # Request stop
        state_handler.request_stop("User clicked stop button")
        
        # Update UI immediately
        _update_ui_for_stop(ui_components, ui_logger)
        
        # Notify service if available
        _notify_service_stop(ui_components, ui_logger)
        
        ui_logger.info("✅ Proses augmentasi berhasil dihentikan")
        
    except Exception as e:
        ui_logger.error(f"❌ Error saat menghentikan augmentasi: {str(e)}")

def stop_augmentation_process(ui_components: Dict[str, Any]) -> None:
    """
    Programmatic stop untuk proses augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    ui_logger = create_ui_logger_bridge(ui_components, "stop_handler")
    state_handler = StateHandler(ui_components, ui_logger)
    
    # Set stop flags
    state_handler.request_stop("Programmatic stop")
    
    # Notify service
    _notify_service_stop(ui_components, ui_logger)
    
    ui_logger.warning("⏹️ Permintaan stop augmentasi diterima")

def _update_ui_for_stop(ui_components: Dict[str, Any], ui_logger) -> None:
    """Update UI saat stop ditekan."""
    # Update status panel
    _update_status_panel(ui_components, "Menghentikan augmentasi...", "warning")
    
    # Sembunyikan tombol stop, tampilkan tombol augment
    if 'stop_button' in ui_components:
        stop_button = ui_components['stop_button']
        if hasattr(stop_button, 'layout'):
            stop_button.layout.display = 'none'
    
    if 'augment_button' in ui_components:
        augment_button = ui_components['augment_button']
        if hasattr(augment_button, 'layout'):
            augment_button.layout.display = 'block'
        if hasattr(augment_button, 'disabled'):
            augment_button.disabled = False
    
    # Enable tombol lainnya
    _enable_other_buttons(ui_components)

def _notify_service_stop(ui_components: Dict[str, Any], ui_logger) -> None:
    """Notify service untuk stop processing."""
    # Notify service jika ada
    if 'augmentation_service' in ui_components:
        try:
            service = ui_components['augmentation_service']
            if hasattr(service, 'stop_processing'):
                service.stop_processing()
            elif hasattr(service, '_stop_signal'):
                service._stop_signal = True
        except Exception as e:
            ui_logger.warning(f"⚠️ Gagal menghentikan service: {str(e)}")

def _enable_other_buttons(ui_components: Dict[str, Any]) -> None:
    """Enable tombol-tombol lainnya."""
    button_keys = ['reset_button', 'cleanup_button', 'save_button']
    
    for key in button_keys:
        if key in ui_components:
            button = ui_components[key]
            if hasattr(button, 'disabled'):
                button.disabled = False

def _update_status_panel(ui_components: Dict[str, Any], message: str, status: str) -> None:
    """Update status panel jika tersedia."""
    if 'status_panel' in ui_components:
        try:
            from smartcash.ui.components.status_panel import update_status_panel
            update_status_panel(ui_components['status_panel'], message, status)
        except ImportError:
            pass  # Status panel tidak tersedia

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