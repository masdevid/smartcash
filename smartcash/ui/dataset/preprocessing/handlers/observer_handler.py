"""
File: smartcash/ui/dataset/preprocessing/handlers/observer_handler.py
Deskripsi: Handler untuk notifikasi observer pattern pada proses preprocessing
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import ICONS
from smartcash.components.observer import ObserverManager
from smartcash.ui.dataset.preprocessing.utils.notification_manager import (
    PreprocessingUIEvents, 
    PREPROCESSING_LOGGER_NAMESPACE,
    get_observer_manager,
    notify_log,
    notify_progress,
    notify_status
)
from smartcash.ui.dataset.preprocessing.utils.ui_observers import register_ui_observers
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message

def setup_observer_handler(ui_components: Dict[str, Any], observer_group: str = "preprocessing_observers") -> Dict[str, Any]:
    """
    Setup observer handler untuk preprocessing dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        observer_group: Nama grup observer (default: preprocessing_observers)
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Cek apakah observer manager sudah disetup
    if 'observer_manager' in ui_components:
        observer_manager = ui_components['observer_manager']
    else:
        # Gunakan fungsi get_observer_manager dari notification_manager
        try:
            observer_manager = get_observer_manager()
            ui_components['observer_manager'] = observer_manager
        except (ImportError, AttributeError):
            # Fallback ke observer manager lokal jika tidak bisa import
            observer_manager = ObserverManager()
            ui_components['observer_manager'] = observer_manager
    
    # Simpan nama grup observer
    ui_components['observer_group'] = observer_group
    
    # Register UI observers
    register_ui_observers(ui_components)
    
    # Log setup berhasil
    log_message(ui_components, "Observer handler berhasil disetup", "debug", "✅")
    
    return ui_components

def notify_process_start(ui_components: Dict[str, Any], process_name: str, display_info: str, split: Optional[str] = None) -> None:
    """
    Notifikasi observer bahwa proses telah dimulai.
    
    Args:
        ui_components: Dictionary komponen UI
        process_name: Nama proses yang dimulai
        display_info: Informasi tambahan untuk ditampilkan
        split: Split dataset yang diproses (opsional)
    """
    # Log pesan
    log_message(ui_components, f"Memulai {process_name} {display_info}", "info", ICONS['start'])
    
    # Notifikasi melalui observer
    if 'observer_manager' in ui_components:
        ui_components['observer_manager'].notify(
            PreprocessingUIEvents.PREPROCESS_START,
            ui_components,
            process_name=process_name,
            display_info=display_info,
            split=split,
            namespace=PREPROCESSING_LOGGER_NAMESPACE
        )
    
    # Update status panel
    notify_status(ui_components, "started", f"Memulai {process_name} {display_info}")
    
    # Panggil callback jika tersedia (untuk kompatibilitas dengan kode lama)
    if 'on_process_start' in ui_components and callable(ui_components['on_process_start']):
        ui_components['on_process_start']("preprocessing", {
            'split': split,
            'display_info': display_info
        })

def notify_process_complete(ui_components: Dict[str, Any], result: Dict[str, Any], display_info: str) -> None:
    """
    Notifikasi observer bahwa proses telah selesai dengan sukses.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Dictionary hasil proses
        display_info: Informasi tambahan untuk ditampilkan
    """
    # Log pesan
    log_message(ui_components, f"Preprocessing {display_info} selesai", "success", ICONS['success'])
    
    # Notifikasi melalui observer
    if 'observer_manager' in ui_components:
        ui_components['observer_manager'].notify(
            PreprocessingUIEvents.PREPROCESS_END,
            ui_components,
            result=result,
            display_info=display_info,
            namespace=PREPROCESSING_LOGGER_NAMESPACE
        )
    
    # Update status panel
    notify_status(ui_components, "completed", f"Preprocessing {display_info} selesai")
    
    # Panggil callback jika tersedia (untuk kompatibilitas dengan kode lama)
    if 'on_process_complete' in ui_components and callable(ui_components['on_process_complete']):
        ui_components['on_process_complete']("preprocessing", result)

def notify_process_error(ui_components: Dict[str, Any], error_message: str) -> None:
    """
    Notifikasi observer bahwa proses mengalami error.
    
    Args:
        ui_components: Dictionary komponen UI
        error_message: Pesan error yang terjadi
    """
    # Log pesan
    log_message(ui_components, f"Error pada preprocessing: {error_message}", "error", ICONS['error'])
    
    # Notifikasi melalui observer
    if 'observer_manager' in ui_components:
        ui_components['observer_manager'].notify(
            PreprocessingUIEvents.PREPROCESS_ERROR,
            ui_components,
            error_message=error_message,
            namespace=PREPROCESSING_LOGGER_NAMESPACE
        )
    
    # Update status panel
    notify_status(ui_components, "failed", f"Error: {error_message}")
    
    # Panggil callback jika tersedia (untuk kompatibilitas dengan kode lama)
    if 'on_process_error' in ui_components and callable(ui_components['on_process_error']):
        ui_components['on_process_error']("preprocessing", error_message)

def notify_process_progress(ui_components: Dict[str, Any], progress: int, total: int = 100, message: str = "") -> None:
    """
    Notifikasi observer tentang progress preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        progress: Nilai progress saat ini
        total: Total nilai progress
        message: Pesan progress
    """
    # Skip jika ui_components tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Gunakan fungsi notify_progress dari notification_manager
    notify_progress(ui_components, progress, total, message)
