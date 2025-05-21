"""
File: smartcash/ui/dataset/preprocessing/utils/ui_observers.py
Deskripsi: Utilitas untuk observer UI pada modul preprocessing dataset
"""

from typing import Dict, Any, Callable, Optional
from smartcash.ui.utils.constants import ICONS

def register_ui_observers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Register observer untuk UI preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Import observer
    try:
        from smartcash.common.observer import ObserverManager, EventType
        
        # Cek apakah observer manager sudah ada
        if 'observer_manager' not in ui_components:
            ui_components['observer_manager'] = ObserverManager()
        
        # Get observer manager
        observer_manager = ui_components['observer_manager']
        
        # Register event handlers
        register_progress_handlers(ui_components, observer_manager)
        register_status_handlers(ui_components, observer_manager)
        register_message_handlers(ui_components, observer_manager)
        register_completion_handlers(ui_components, observer_manager)
        
        # Setup observer group untuk UI
        ui_components['observer_group'] = "preprocessing_ui"
        
    except ImportError:
        # Log warning jika observer tidak tersedia
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message']("ObserverManager tidak tersedia. Fitur notifikasi dibatasi.", "warning", "⚠️")
    
    return ui_components

def register_progress_handlers(ui_components: Dict[str, Any], observer_manager: Any) -> None:
    """
    Register handler untuk event progress.
    
    Args:
        ui_components: Dictionary komponen UI
        observer_manager: Observer manager
    """
    from smartcash.common.observer import EventType
    
    # Handler untuk progress update
    def handle_progress_update(data: Dict[str, Any]) -> None:
        # Pastikan data valid
        if not isinstance(data, dict):
            return
        
        # Extract data
        value = data.get('value', 0)
        max_value = data.get('max_value', 100)
        message = data.get('message', '')
        
        # Update progress UI
        from smartcash.ui.dataset.preprocessing.utils.progress_manager import update_progress
        update_progress(ui_components, value, max_value, message)
    
    # Register observer untuk progress update
    observer_manager.register(
        event_type=EventType.PROGRESS_UPDATE,
        callback=handle_progress_update,
        group="preprocessing_ui"
    )

def register_status_handlers(ui_components: Dict[str, Any], observer_manager: Any) -> None:
    """
    Register handler untuk event status.
    
    Args:
        ui_components: Dictionary komponen UI
        observer_manager: Observer manager
    """
    from smartcash.common.observer import EventType
    
    # Handler untuk status update
    def handle_status_update(data: Dict[str, Any]) -> None:
        # Pastikan data valid
        if not isinstance(data, dict):
            return
        
        # Extract data
        status = data.get('status', 'idle')
        message = data.get('message', '')
        
        # Update status UI
        from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import update_status_panel
        update_status_panel(ui_components, status, message)
    
    # Register observer untuk status update
    observer_manager.register(
        event_type=EventType.STATUS_UPDATE,
        callback=handle_status_update,
        group="preprocessing_ui"
    )

def register_message_handlers(ui_components: Dict[str, Any], observer_manager: Any) -> None:
    """
    Register handler untuk event message/log.
    
    Args:
        ui_components: Dictionary komponen UI
        observer_manager: Observer manager
    """
    from smartcash.common.observer import EventType
    
    # Handler untuk log message
    def handle_log_message(data: Dict[str, Any]) -> None:
        # Pastikan data valid
        if not isinstance(data, dict):
            return
        
        # Extract data
        message = data.get('message', '')
        level = data.get('level', 'info')
        icon = data.get('icon', '')
        
        # Log message
        from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
        log_message(ui_components, message, level, icon)
    
    # Register observer untuk log message
    observer_manager.register(
        event_type=EventType.LOG_MESSAGE,
        callback=handle_log_message,
        group="preprocessing_ui"
    )

def register_completion_handlers(ui_components: Dict[str, Any], observer_manager: Any) -> None:
    """
    Register handler untuk event completion.
    
    Args:
        ui_components: Dictionary komponen UI
        observer_manager: Observer manager
    """
    from smartcash.common.observer import EventType
    
    # Handler untuk completion
    def handle_completion(data: Dict[str, Any]) -> None:
        # Pastikan data valid
        if not isinstance(data, dict):
            return
        
        # Extract data
        success = data.get('success', False)
        message = data.get('message', '')
        
        # Update UI berdasarkan hasil
        from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import update_ui_state, reset_ui_after_preprocessing
        
        if success:
            update_ui_state(ui_components, 'success', message or "Preprocessing berhasil diselesaikan")
        else:
            update_ui_state(ui_components, 'error', message or "Terjadi kesalahan saat preprocessing")
        
        # Reset UI state
        reset_ui_after_preprocessing(ui_components)
    
    # Register observer untuk completion
    observer_manager.register(
        event_type=EventType.TASK_COMPLETED,
        callback=handle_completion,
        group="preprocessing_ui"
    )

def notify_process_start(ui_components: Dict[str, Any], process_name: str, display_info: str, split: Optional[str] = None) -> None:
    """
    Notifikasi observer bahwa proses telah dimulai.
    
    Args:
        ui_components: Dictionary komponen UI
        process_name: Nama proses yang dimulai
        display_info: Informasi tambahan untuk ditampilkan
        split: Split dataset yang diproses (opsional)
    """
    logger = ui_components.get('logger')
    if logger: 
        logger.info(f"{ICONS['start']} Memulai {process_name} {display_info}")
    
    # Panggil callback jika tersedia
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
        result: Hasil dari proses
        display_info: Informasi tambahan untuk ditampilkan
    """
    logger = ui_components.get('logger')
    if logger: 
        logger.info(f"{ICONS['success']} Preprocessing {display_info} selesai")
    
    # Panggil callback jika tersedia
    if 'on_process_complete' in ui_components and callable(ui_components['on_process_complete']):
        ui_components['on_process_complete']("preprocessing", result)

def notify_process_error(ui_components: Dict[str, Any], error_message: str) -> None:
    """
    Notifikasi observer bahwa proses mengalami error.
    
    Args:
        ui_components: Dictionary komponen UI
        error_message: Pesan error
    """
    logger = ui_components.get('logger')
    if logger: 
        logger.error(f"{ICONS['error']} Error pada preprocessing: {error_message}")
    
    # Panggil callback jika tersedia
    if 'on_process_error' in ui_components and callable(ui_components['on_process_error']):
        ui_components['on_process_error']("preprocessing", error_message)

def notify_process_stop(ui_components: Dict[str, Any], display_info: str = "") -> None:
    """
    Notifikasi observer bahwa proses telah dihentikan oleh pengguna.
    
    Args:
        ui_components: Dictionary komponen UI
        display_info: Informasi tambahan untuk ditampilkan
    """
    logger = ui_components.get('logger')
    if logger: 
        logger.warning(f"{ICONS['stop']} Proses preprocessing dihentikan oleh pengguna")
    
    # Panggil callback jika tersedia
    if 'on_process_stop' in ui_components and callable(ui_components['on_process_stop']):
        ui_components['on_process_stop']("preprocessing", {
            'display_info': display_info
        })

def disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True) -> None:
    """
    Menonaktifkan atau mengaktifkan komponen UI selama proses berjalan.
    
    Args:
        ui_components: Dictionary komponen UI
        disable: True untuk menonaktifkan, False untuk mengaktifkan
    """
    # Daftar komponen yang perlu dinonaktifkan
    disable_components = [
        'split_selector', 'config_accordion', 'options_accordion',
        'reset_button', 'preprocess_button', 'save_button'
    ]
    
    # Disable/enable komponen
    for component in disable_components:
        if component in ui_components:
            widget = ui_components[component]
            if hasattr(widget, 'disabled'):
                widget.disabled = disable
            elif hasattr(widget, 'layout'):
                # For widgets without disabled attribute, use opacity
                if not hasattr(widget.layout, 'opacity'):
                    # Create new layout with opacity
                    new_layout = type(widget.layout)()
                    for key, value in widget.layout.trait_values().items():
                        setattr(new_layout, key, value)
                    widget.layout = new_layout
                widget.layout.opacity = '0.5' if disable else '1'
