"""
File: smartcash/ui/dataset/augmentation/handlers/notification_handler.py
Deskripsi: Handler notifikasi untuk augmentasi dataset
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

def notify_process_start(ui_components: Dict[str, Any], process_name: str = "augmentasi", display_info: str = "", split: Optional[str] = None) -> None:
    """
    Notifikasi bahwa proses augmentasi dimulai.
    
    Args:
        ui_components: Dictionary komponen UI
        process_name: Nama proses yang dimulai
        display_info: Informasi tambahan untuk ditampilkan
        split: Split dataset yang diproses (opsional)
    """
    logger = ui_components.get('logger', get_logger())
    if logger: logger.info(f"{ICONS['start']} Memulai {process_name} {display_info}")
    
    # Panggil callback jika tersedia
    if 'on_process_start' in ui_components and callable(ui_components['on_process_start']):
        ui_components['on_process_start']("augmentation", {
            'split': split,
            'display_info': display_info
        })
    
    # Notifikasi melalui observer pattern
    try:
        from smartcash.components.observer import notify
        notify('augmentation_started', {
            'ui_components': ui_components,
            'split': split,
            'display_info': display_info
        })
    except Exception as e:
        logger.debug(f"Gagal mengirim notifikasi augmentation_started: {str(e)}")

def notify_process_stop(ui_components: Dict[str, Any], display_info: str = "") -> None:
    """
    Notifikasi bahwa proses augmentasi dihentikan.
    
    Args:
        ui_components: Dictionary komponen UI
        display_info: Informasi tambahan untuk ditampilkan
    """
    logger = ui_components.get('logger', get_logger())
    if logger: logger.info(f"{ICONS['stop']} Augmentasi {display_info} dihentikan")
    
    # Panggil callback jika tersedia
    if 'on_process_stop' in ui_components and callable(ui_components['on_process_stop']):
        ui_components['on_process_stop']("augmentation", {
            'display_info': display_info
        })
    
    # Notifikasi melalui observer pattern
    try:
        from smartcash.components.observer import notify
        notify('augmentation_stopped', {
            'ui_components': ui_components,
            'display_info': display_info
        })
    except Exception as e:
        logger.debug(f"Gagal mengirim notifikasi augmentation_stopped: {str(e)}")

def notify_process_complete(ui_components: Dict[str, Any], result: Optional[Dict[str, Any]] = None, display_info: str = "") -> None:
    """
    Notifikasi bahwa proses augmentasi selesai.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil augmentasi
        display_info: Informasi tambahan untuk ditampilkan
    """
    logger = ui_components.get('logger', get_logger())
    if logger: logger.info(f"{ICONS['success']} Augmentasi {display_info} selesai")
    
    # Panggil callback jika tersedia
    if 'on_process_complete' in ui_components and callable(ui_components['on_process_complete']):
        ui_components['on_process_complete']("augmentation", result)
    
    # Notifikasi melalui observer pattern
    try:
        from smartcash.components.observer import notify
        notify('augmentation_completed', {
            'ui_components': ui_components, 
            'result': result,
            'display_info': display_info
        })
    except Exception as e:
        logger.debug(f"Gagal mengirim notifikasi augmentation_completed: {str(e)}")

def notify_process_error(ui_components: Dict[str, Any], error_message: str) -> None:
    """
    Notifikasi bahwa proses augmentasi mengalami error.
    
    Args:
        ui_components: Dictionary komponen UI
        error_message: Pesan error yang terjadi
    """
    logger = ui_components.get('logger', get_logger())
    if logger: logger.error(f"{ICONS['error']} Error pada augmentasi: {error_message}")
    
    # Panggil callback jika tersedia
    if 'on_process_error' in ui_components and callable(ui_components['on_process_error']):
        ui_components['on_process_error']("augmentation", error_message)
    
    # Notifikasi melalui observer pattern
    try:
        from smartcash.components.observer import notify
        notify('augmentation_error', {
            'ui_components': ui_components,
            'error': error_message
        })
    except Exception as e:
        logger.debug(f"Gagal mengirim notifikasi augmentation_error: {str(e)}")
