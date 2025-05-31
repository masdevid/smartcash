"""
File: smartcash/ui/pretrained_model/handlers/setup_handlers.py
Deskripsi: Setup handlers untuk model pretrained dengan pendekatan DRY
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

logger = get_logger(__name__)

def setup_model_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handlers untuk model pretrained dengan progress tracking - one-liner style.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (opsional)
        config: Konfigurasi (opsional)
        
    Returns:
        Dictionary komponen UI yang diperbarui
    """
    # Import handler yang diperlukan
    from smartcash.ui.pretrained_model.handlers.ui_handlers import handle_download_sync_button, handle_reset_ui_button
    
    # Register handler untuk tombol download & sync - one-liner style
    if 'download_sync_button' in ui_components and ui_components['download_sync_button']:
        ui_components['download_sync_button'].on_click(lambda b: handle_download_sync_button(b, ui_components))
        logger.debug(f"{ICONS.get('link', '🔗')} Handler untuk tombol download & sync terdaftar")
    
    # Register handler untuk tombol reset UI
    if 'reset_ui_button' in ui_components and ui_components['reset_ui_button']:
        ui_components['reset_ui_button'].on_click(lambda b: handle_reset_ui_button(b, ui_components))
        logger.debug(f"{ICONS.get('link', '🔗')} Handler untuk tombol reset UI terdaftar")
    
    # Definisikan fungsi reset_progress_bar untuk progress tracking dengan parameter show_progress
    ui_components['reset_progress_bar'] = lambda value=0, message="", show_progress=True: reset_progress_bar(ui_components, value, message, show_progress)
    
    # Definisikan fungsi log_message untuk digunakan di berbagai tempat
    from smartcash.ui.pretrained_model.utils.logger_utils import log_message
    ui_components['log_message'] = lambda message, level="info": log_message(ui_components, message, level)
    
    return ui_components

# Fungsi utilitas untuk progress tracking - one-liner style
def reset_progress_bar(ui_components: Dict[str, Any], value: int = 0, message: str = "", show_progress: bool = True) -> None:
    """Reset progress bar dan label dengan nilai dan pesan awal"""
    # Untuk test case, selalu update nilai langsung
    if 'progress_bar' in ui_components:
        # Selalu set nilai langsung untuk kompatibilitas dengan test
        ui_components['progress_bar'].value = value
        ui_components['progress_bar'].max = 100
        
        # Atur visibilitas progress bar
        if hasattr(ui_components['progress_bar'], 'layout'):
            ui_components['progress_bar'].layout.visibility = 'visible' if show_progress else 'hidden'
        
        # Jika API baru tersedia, gunakan juga
        if hasattr(ui_components['progress_bar'], 'reset') and value == 0:
            ui_components['progress_bar'].reset()
    
    # Gunakan API reset_all jika tersedia (API baru) dan nilai adalah 0
    if 'reset_all' in ui_components and callable(ui_components['reset_all']) and value == 0:
        ui_components['reset_all']()
        
        # Atur visibilitas tracker jika tersedia
        if 'tracker' in ui_components:
            if hasattr(ui_components['tracker'], 'show') and hasattr(ui_components['tracker'], 'hide'):
                if show_progress:
                    ui_components['tracker'].show()
                else:
                    ui_components['tracker'].hide()
    
    # Reset progress label jika tersedia (API lama)
    if 'progress_label' in ui_components and hasattr(ui_components['progress_label'], 'value'):
        ui_components['progress_label'].value = message or "Siap"
        
        # Pastikan label selalu terlihat meskipun progress bar disembunyikan
        if hasattr(ui_components['progress_label'], 'layout'):
            ui_components['progress_label'].layout.visibility = 'visible'
        
    # Update status widget jika tersedia (API baru)
    if 'status_widget' in ui_components and hasattr(ui_components['status_widget'], 'value'):
        ui_components['status_widget'].value = message or "Siap"
        
        # Pastikan status widget selalu terlihat
        if hasattr(ui_components['status_widget'], 'layout'):
            ui_components['status_widget'].layout.visibility = 'visible'
        
    # Update progress jika nilai bukan 0 dan API update tersedia
    if value != 0 and 'update_progress' in ui_components and callable(ui_components['update_progress']):
        ui_components['update_progress'](value, message)

def setup_model_cleanup_handler(ui_components: Dict[str, Any], module_name: str = None, config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """
    Setup cleanup handler untuk model pretrained (minimal implementation).
    
    Args:
        ui_components: Dictionary komponen UI
        module_name: Nama modul (opsional)
        config: Konfigurasi (opsional)
        env: Environment manager (opsional)
        
    Returns:
        Dictionary komponen UI yang diperbarui
    """
    # Model tidak memerlukan cleanup handler yang kompleks
    ui_components['cleanup'] = lambda: logger.info(f"{ICONS.get('cleanup', '🧹')} Model cleanup tidak diperlukan")
    
    return ui_components
