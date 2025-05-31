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
    from smartcash.ui.pretrained_model.handlers.ui_handlers import handle_download_sync_button
    
    # Register handler untuk tombol download & sync - one-liner style
    if 'download_sync_button' in ui_components and ui_components['download_sync_button']:
        ui_components['download_sync_button'].on_click(lambda b: handle_download_sync_button(b, ui_components))
        logger.debug(f"{ICONS.get('link', '🔗')} Handler untuk tombol download & sync terdaftar")
    
    # Definisikan fungsi reset_progress_bar untuk progress tracking - one-liner style
    ui_components['reset_progress_bar'] = lambda value=0, message="": reset_progress_bar(ui_components, value, message)
    
    return ui_components

# Fungsi utilitas untuk progress tracking - one-liner style
def reset_progress_bar(ui_components: Dict[str, Any], value: int = 0, message: str = "") -> None:
    """Reset progress bar dan label dengan nilai dan pesan awal - one-liner style"""
    # Untuk test case, selalu update nilai langsung
    if 'progress_bar' in ui_components:
        # Selalu set nilai langsung untuk kompatibilitas dengan test
        ui_components['progress_bar'].value = value
        ui_components['progress_bar'].max = 100
        
        # Jika API baru tersedia, gunakan juga
        if hasattr(ui_components['progress_bar'], 'reset') and value == 0:
            ui_components['progress_bar'].reset()
    
    # Gunakan API reset_all jika tersedia (API baru) dan nilai adalah 0
    if 'reset_all' in ui_components and callable(ui_components['reset_all']) and value == 0:
        ui_components['reset_all']()
    
    # Reset progress label jika tersedia (API lama)
    if 'progress_label' in ui_components and hasattr(ui_components['progress_label'], 'value'):
        ui_components['progress_label'].value = message or "Siap"
        
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
