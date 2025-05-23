"""
File: smartcash/ui/dataset/download/utils/notification_manager.py
Deskripsi: Utilitas untuk mengelola notifikasi UI pada proses download dataset
"""

from typing import Dict, Any, Optional
from smartcash.components.observer import notify, EventTopics, ObserverManager
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
    
    # Event untuk step progress
    STEP_PROGRESS_START = "DOWNLOAD_STEP_PROGRESS_START"
    STEP_PROGRESS_UPDATE = "DOWNLOAD_STEP_PROGRESS_UPDATE"
    STEP_PROGRESS_COMPLETE = "DOWNLOAD_STEP_PROGRESS_COMPLETE"
    STEP_PROGRESS_ERROR = "DOWNLOAD_STEP_PROGRESS_ERROR"

# Singleton observer manager untuk digunakan oleh semua komponen
_OBSERVER_MANAGER = None

def get_observer_manager() -> ObserverManager:
    """
    Dapatkan instance ObserverManager yang digunakan untuk notifikasi download.
    Ini menggunakan pola singleton untuk memastikan hanya satu instance yang digunakan.
    
    Returns:
        ObserverManager: Instance observer manager
    """
    global _OBSERVER_MANAGER
    
    if _OBSERVER_MANAGER is None:
        _OBSERVER_MANAGER = ObserverManager()
    
    return _OBSERVER_MANAGER

def notify_log(
    sender: Any,
    message: str,
    level: str = "info",
    observer_manager=None,
    namespace: str = "dataset.download",
    **kwargs
) -> None:
    """
    Mengirim notifikasi ke log output UI
    
    Args:
        sender: Objek pengirim notifikasi
        message: Pesan yang akan ditampilkan
        level: Level log (info, warning, error, success)
        observer_manager: Observer manager opsional
        namespace: Namespace logger untuk memisahkan log dari modul lain
        **kwargs: Parameter tambahan
    """
    logger = get_logger(namespace)
    
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
        params = {"message": message, "level": level, "namespace": namespace}
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
    namespace: str = "dataset.download",
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
        namespace: Namespace logger untuk memisahkan log dari modul lain
        **kwargs: Parameter tambahan
    """
    logger = get_logger(namespace)
    
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
        try:
            percentage = min(100, max(0, int((float(progress) / float(total)) * 100)))
        except (ValueError, TypeError):
            percentage = 0
    
    # Kirim notifikasi ke UI
    try:
        params = {"namespace": namespace}
        if progress is not None:
            try:
                params["progress"] = int(float(progress))
            except (ValueError, TypeError):
                params["progress"] = 0
        if total is not None:
            try:
                params["total"] = int(float(total))
            except (ValueError, TypeError):
                params["total"] = 100
        if percentage is not None:
            params["percentage"] = percentage
        if message is not None:
            params["message"] = message
        
        # Tambahkan parameter untuk step progress jika ada
        if "step" in kwargs:
            step_event_mapping = {
                "start": DownloadUIEvents.STEP_PROGRESS_START,
                "update": DownloadUIEvents.STEP_PROGRESS_UPDATE,
                "complete": DownloadUIEvents.STEP_PROGRESS_COMPLETE,
                "error": DownloadUIEvents.STEP_PROGRESS_ERROR
            }
            step_event = step_event_mapping.get(event_type.lower(), DownloadUIEvents.STEP_PROGRESS_UPDATE)
            
            # Kirim notifikasi step progress
            step_params = {
                "step": kwargs["step"],
                "step_message": kwargs.get("step_message", message or ""),
                "step_progress": params.get("progress", 0),
                "step_total": params.get("total", 100),
                "total_steps": kwargs.get("total_steps", 5),
                "current_step": kwargs.get("current_step", 1),
                "namespace": namespace
            }
            
            if observer_manager is not None:
                observer_manager.notify(step_event, sender, **step_params)
            else:
                notify(step_event, sender, **step_params)
        
        params.update(kwargs)
        
        if observer_manager is not None:
            observer_manager.notify(ui_event, sender, **params)
        else:
            notify(ui_event, sender, **params)
    except Exception as e:
        logger.error(f"❌ Gagal mengirim notifikasi progress: {str(e)}")
