"""
File: smartcash/ui/dataset/preprocessing/utils/notification_manager.py
Deskripsi: Utilitas untuk mengelola notifikasi dan event pada proses preprocessing dataset
"""

from enum import Enum, auto
from typing import Dict, Any, Optional, Callable
from IPython.display import display
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.components.observer import ObserverManager

# Konstanta untuk namespace logger
PREPROCESSING_LOGGER_NAMESPACE = "smartcash.dataset.preprocessing"
# Konstanta untuk ID namespace di UI
MODULE_LOGGER_NAME = "DATASET-PREPROCESSING"

class PreprocessingUIEvents(str, Enum):
    """Event types untuk UI preprocessing dataset."""
    # Log events
    LOG_INFO = "preprocessing_log_info"
    LOG_WARNING = "preprocessing_log_warning"
    LOG_ERROR = "preprocessing_log_error"
    LOG_SUCCESS = "preprocessing_log_success"
    LOG_DEBUG = "preprocessing_log_debug"
    
    # Progress events
    PROGRESS_START = "preprocessing_progress_start"
    PROGRESS_UPDATE = "preprocessing_progress_update"
    PROGRESS_COMPLETE = "preprocessing_progress_complete"
    PROGRESS_ERROR = "preprocessing_progress_error"
    
    # Step Progress events
    STEP_PROGRESS_START = "preprocessing_step_progress_start"
    STEP_PROGRESS_UPDATE = "preprocessing_step_progress_update"
    STEP_PROGRESS_COMPLETE = "preprocessing_step_progress_complete"
    STEP_PROGRESS_ERROR = "preprocessing_step_progress_error"
    
    # Status events
    STATUS_STARTED = "preprocessing_status_started"
    STATUS_COMPLETED = "preprocessing_status_completed"
    STATUS_FAILED = "preprocessing_status_failed"
    STATUS_PAUSED = "preprocessing_status_paused"
    STATUS_RESUMED = "preprocessing_status_resumed"
    
    # Config events
    CONFIG_UPDATED = "preprocessing_config_updated"
    CONFIG_SAVED = "preprocessing_config_saved"
    CONFIG_LOADED = "preprocessing_config_loaded"
    CONFIG_RESET = "preprocessing_config_reset"
    
    # Preprocessing-specific events
    PREPROCESS_START = "preprocessing_start"
    PREPROCESS_END = "preprocessing_end"
    PREPROCESS_ERROR = "preprocessing_error"
    PREPROCESS_PROGRESS = "preprocessing_progress"
    
    # Cleanup events
    CLEANUP_START = "preprocessing_cleanup_start"
    CLEANUP_END = "preprocessing_cleanup_end"
    CLEANUP_ERROR = "preprocessing_cleanup_error"
    
    # Reset events
    RESET_START = "preprocessing_reset_start"
    RESET_END = "preprocessing_reset_end"
    
    # Save events
    SAVE_START = "preprocessing_save_start"
    SAVE_END = "preprocessing_save_end"
    SAVE_ERROR = "preprocessing_save_error"

class ObserverGroupNames(str, Enum):
    """Nama group untuk observer."""
    PREPROCESSING = "preprocessing_observers"
    LOG = "preprocessing_log_observers"
    PROGRESS = "preprocessing_progress_observers"
    STATUS = "preprocessing_status_observers"
    CONFIG = "preprocessing_config_observers"

# Singleton instance untuk ObserverManager
_observer_manager_instance: Optional[ObserverManager] = None

def get_observer_manager() -> ObserverManager:
    """
    Get singleton instance dari ObserverManager.
    
    Returns:
        ObserverManager: Instance dari ObserverManager.
    """
    global _observer_manager_instance
    
    if _observer_manager_instance is None:
        _observer_manager_instance = ObserverManager()
    
    return _observer_manager_instance

def notify_progress(ui_components: Dict[str, Any], progress: int, total: int = 100, message: str = "") -> None:
    """
    Notifikasi progress update melalui observer system.
    
    Args:
        ui_components: Dictionary komponen UI
        progress: Nilai progress saat ini
        total: Total nilai progress
        message: Pesan progress
    """
    if 'observer_manager' not in ui_components:
        return
    
    observer_manager = ui_components['observer_manager']
    
    # Notifikasi progress event
    observer_manager.notify(
        PreprocessingUIEvents.PROGRESS_UPDATE,
        ui_components,
        progress=progress,
        total=total,
        message=message,
        namespace=PREPROCESSING_LOGGER_NAMESPACE
    )

def notify_step_progress(ui_components: Dict[str, Any], step_progress: int, step_total: int = 100, 
                        step_message: str = "", current_step: int = 1, total_steps: int = 1) -> None:
    """
    Notifikasi step progress update melalui observer system.
    
    Args:
        ui_components: Dictionary komponen UI
        step_progress: Nilai progress step saat ini
        step_total: Total nilai progress step
        step_message: Pesan step progress
        current_step: Langkah saat ini
        total_steps: Total langkah
    """
    if 'observer_manager' not in ui_components:
        return
    
    observer_manager = ui_components['observer_manager']
    
    # Notifikasi step progress event
    observer_manager.notify(
        PreprocessingUIEvents.STEP_PROGRESS_UPDATE,
        ui_components,
        step_progress=step_progress,
        step_total=step_total,
        step_message=step_message,
        current_step=current_step,
        total_steps=total_steps,
        namespace=PREPROCESSING_LOGGER_NAMESPACE
    )

def notify_log(ui_components: Dict[str, Any], message: str, level: str = "info", icon: str = "ℹ️") -> None:
    """
    Notifikasi log message melalui observer system.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan log
        level: Level log (info, warning, error, success)
        icon: Icon untuk pesan log
    """
    if 'observer_manager' not in ui_components:
        return
    
    observer_manager = ui_components['observer_manager']
    
    # Map level ke event type
    level_to_event = {
        "info": PreprocessingUIEvents.LOG_INFO,
        "warning": PreprocessingUIEvents.LOG_WARNING,
        "error": PreprocessingUIEvents.LOG_ERROR,
        "success": PreprocessingUIEvents.LOG_SUCCESS,
        "debug": PreprocessingUIEvents.LOG_DEBUG
    }
    
    event_type = level_to_event.get(level.lower(), PreprocessingUIEvents.LOG_INFO)
    
    # Notifikasi log event
    observer_manager.notify(
        event_type,
        ui_components,
        message=message,
        level=level,
        icon=icon,
        namespace=PREPROCESSING_LOGGER_NAMESPACE
    )

def notify_status(ui_components: Dict[str, Any], status: str, message: str = "") -> None:
    """
    Notifikasi status update melalui observer system.
    
    Args:
        ui_components: Dictionary komponen UI
        status: Status baru
        message: Pesan status
    """
    if 'observer_manager' not in ui_components:
        return
    
    observer_manager = ui_components['observer_manager']
    
    # Map status ke event type
    status_to_event = {
        "started": PreprocessingUIEvents.STATUS_STARTED,
        "completed": PreprocessingUIEvents.STATUS_COMPLETED,
        "failed": PreprocessingUIEvents.STATUS_FAILED,
        "paused": PreprocessingUIEvents.STATUS_PAUSED,
        "resumed": PreprocessingUIEvents.STATUS_RESUMED
    }
    
    event_type = status_to_event.get(status.lower(), PreprocessingUIEvents.STATUS_UPDATED)
    
    # Notifikasi status event
    observer_manager.notify(
        event_type,
        ui_components,
        status=status,
        message=message,
        namespace=PREPROCESSING_LOGGER_NAMESPACE
    )

def notify_config(ui_components: Dict[str, Any], action: str, config: Dict[str, Any]) -> None:
    """
    Notifikasi config update melalui observer system.
    
    Args:
        ui_components: Dictionary komponen UI
        action: Aksi config (updated, saved, loaded, reset)
        config: Data konfigurasi
    """
    if 'observer_manager' not in ui_components:
        return
    
    observer_manager = ui_components['observer_manager']
    
    # Map action ke event type
    action_to_event = {
        "updated": PreprocessingUIEvents.CONFIG_UPDATED,
        "saved": PreprocessingUIEvents.CONFIG_SAVED,
        "loaded": PreprocessingUIEvents.CONFIG_LOADED,
        "reset": PreprocessingUIEvents.CONFIG_RESET
    }
    
    event_type = action_to_event.get(action.lower(), PreprocessingUIEvents.CONFIG_UPDATED)
    
    # Notifikasi config event
    observer_manager.notify(
        event_type,
        ui_components,
        action=action,
        config=config,
        namespace=PREPROCESSING_LOGGER_NAMESPACE
    )

class PreprocessingNotificationManager:
    """
    Manager untuk notifikasi preprocessing dengan pendekatan OOP.
    Mengelola notifikasi dan status selama proses preprocessing.
    """
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi notification manager.
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
        self.status_output = ui_components.get('status')
    
    def update_status(self, status_type: str, message: str) -> None:
        """
        Update status output dengan pesan.
        
        Args:
            status_type: Tipe status (success, info, warning, error)
            message: Pesan yang akan ditampilkan
        """
        if self.status_output:
            with self.status_output:
                display(create_status_indicator(status_type, message))
        
        # Update status panel jika tersedia
        if 'update_status_panel' in self.ui_components and callable(self.ui_components['update_status_panel']):
            self.ui_components['update_status_panel'](self.ui_components, status_type, message)
        
        # Log pesan
        if self.logger:
            log_methods = {
                'success': self.logger.success if hasattr(self.logger, 'success') else self.logger.info,
                'info': self.logger.info,
                'warning': self.logger.warning,
                'error': self.logger.error
            }
            
            icon = ICONS.get(status_type, ICONS['info'])
            log_method = log_methods.get(status_type, self.logger.info)
            log_method(f"{icon} {message}")
    
    def notify_progress(self, step: int, total_steps: int, message: str) -> None:
        """
        Update progress bar dan pesan progress.
        
        Args:
            step: Langkah saat ini
            total_steps: Total langkah
            message: Pesan progress
        """
        # Update progress bar jika tersedia
        if 'update_progress' in self.ui_components and callable(self.ui_components['update_progress']):
            self.ui_components['update_progress'](step, total_steps, message)
        
        # Log pesan progress
        if self.logger:
            self.logger.info(f"{ICONS['processing']} {message} ({step}/{total_steps})")

def get_notification_manager(ui_components: Dict[str, Any]) -> PreprocessingNotificationManager:
    """
    Factory function untuk mendapatkan notification manager.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Instance dari PreprocessingNotificationManager
    """
    # Cek apakah sudah ada instance di ui_components
    if 'notification_manager' in ui_components:
        return ui_components['notification_manager']
    
    # Buat instance baru
    manager = PreprocessingNotificationManager(ui_components)
    
    # Simpan ke ui_components
    ui_components['notification_manager'] = manager
    
    return manager
