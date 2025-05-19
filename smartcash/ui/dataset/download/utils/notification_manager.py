"""
File: smartcash/ui/dataset/download/utils/notification_manager.py
Deskripsi: Utilitas untuk mengelola notifikasi UI pada proses download dataset
"""

from typing import Dict, Any, Optional
from smartcash.components.observer import notify, EventTopics
from smartcash.common.logger import get_logger

# Definisi topic event untuk UI
class DownloadUIEvents:
    # Event untuk log output
    LOG_INFO = "DOWNLOAD_LOG_INFO"
    LOG_WARNING = "DOWNLOAD_LOG_WARNING"
    LOG_ERROR = "DOWNLOAD_LOG_ERROR"
    LOG_SUCCESS = "DOWNLOAD_LOG_SUCCESS"
    
    # Event untuk progress bar
    PROGRESS_START = "DOWNLOAD_PROGRESS_START"
    PROGRESS_UPDATE = "DOWNLOAD_PROGRESS_UPDATE"
    PROGRESS_COMPLETE = "DOWNLOAD_PROGRESS_COMPLETE"
    PROGRESS_ERROR = "DOWNLOAD_PROGRESS_ERROR"

def notify_log(
    sender: Any,
    message: str,
    level: str = "info",
    observer_manager=None,
    **kwargs
) -> None:
    """
    Mengirim notifikasi ke log output UI
    
    Args:
        sender: Objek pengirim notifikasi
        message: Pesan yang akan ditampilkan
        level: Level log (info, warning, error, success)
        observer_manager: Observer manager opsional
        **kwargs: Parameter tambahan
    """
    logger = get_logger("download")
    
    # Map level ke event dan fungsi logger
    level_mapping = {
        "info": (DownloadUIEvents.LOG_INFO, logger.info),
        "warning": (DownloadUIEvents.LOG_WARNING, logger.warning),
        "error": (DownloadUIEvents.LOG_ERROR, logger.error),
        "success": (DownloadUIEvents.LOG_SUCCESS, logger.info)
    }
    
    event_type, log_func = level_mapping.get(level.lower(), (DownloadUIEvents.LOG_INFO, logger.info))
    
    # Log ke console dengan emoji sesuai level
    emoji_map = {
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "❌",
        "success": "✅"
    }
    emoji = emoji_map.get(level.lower(), "ℹ️")
    log_func(f"{emoji} {message}")
    
    # Kirim notifikasi ke UI
    try:
        params = {"message": message, "level": level}
        params.update(kwargs)
        
        if observer_manager is not None:
            observer_manager.notify(event_type, sender, **params)
        else:
            notify(event_type, sender, **params)
    except Exception as e:
        logger.error(f"❌ Gagal mengirim notifikasi log: {str(e)}")

def notify_progress(
    sender: Any,
    event_type: str,
    progress: Optional[float] = None,
    total: Optional[float] = None,
    message: Optional[str] = None,
    observer_manager=None,
    **kwargs
) -> None:
    """
    Mengirim notifikasi ke progress bar UI
    
    Args:
        sender: Objek pengirim notifikasi
        event_type: Tipe event (start, update, complete, error)
        progress: Nilai progress saat ini
        total: Nilai total progress
        message: Pesan yang akan ditampilkan
        observer_manager: Observer manager opsional
        **kwargs: Parameter tambahan
    """
    logger = get_logger("download")
    
    # Map event_type ke event
    event_mapping = {
        "start": DownloadUIEvents.PROGRESS_START,
        "update": DownloadUIEvents.PROGRESS_UPDATE,
        "complete": DownloadUIEvents.PROGRESS_COMPLETE,
        "error": DownloadUIEvents.PROGRESS_ERROR
    }
    
    ui_event = event_mapping.get(event_type.lower(), DownloadUIEvents.PROGRESS_UPDATE)
    
    # Hitung persentase jika progress dan total tersedia
    percentage = None
    if progress is not None and total is not None and total > 0:
        percentage = min(100, max(0, int((progress / total) * 100)))
    
    # Kirim notifikasi ke UI
    try:
        params = {}
        if progress is not None:
            params["progress"] = progress
        if total is not None:
            params["total"] = total
        if percentage is not None:
            params["percentage"] = percentage
        if message is not None:
            params["message"] = message
        
        params.update(kwargs)
        
        if observer_manager is not None:
            observer_manager.notify(ui_event, sender, **params)
        else:
            notify(ui_event, sender, **params)
    except Exception as e:
        logger.error(f"❌ Gagal mengirim notifikasi progress: {str(e)}")
