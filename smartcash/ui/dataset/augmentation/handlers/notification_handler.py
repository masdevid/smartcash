"""
File: smartcash/ui/dataset/augmentation/handlers/notification_handler.py
Deskripsi: Handler notifikasi untuk augmentasi dataset
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger

def notify_process_start(ui_components: Dict[str, Any]) -> None:
    """
    Notifikasi bahwa proses augmentasi dimulai.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger', get_logger('augmentation'))
    
    try:
        from smartcash.components.observer import notify
        notify('augmentation_started', {'ui_components': ui_components})
    except Exception as e:
        logger.debug(f"Gagal mengirim notifikasi augmentasi_started: {str(e)}")

def notify_process_stop(ui_components: Dict[str, Any]) -> None:
    """
    Notifikasi bahwa proses augmentasi dihentikan.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger', get_logger('augmentation'))
    
    try:
        from smartcash.components.observer import notify
        notify('augmentation_stopped', {'ui_components': ui_components})
    except Exception as e:
        logger.debug(f"Gagal mengirim notifikasi augmentasi_stopped: {str(e)}")

def notify_process_complete(ui_components: Dict[str, Any], result: Optional[Dict[str, Any]] = None) -> None:
    """
    Notifikasi bahwa proses augmentasi selesai.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil augmentasi
    """
    logger = ui_components.get('logger', get_logger('augmentation'))
    
    try:
        from smartcash.components.observer import notify
        notify('augmentation_completed', {'ui_components': ui_components, 'result': result})
    except Exception as e:
        logger.debug(f"Gagal mengirim notifikasi augmentasi_completed: {str(e)}")
