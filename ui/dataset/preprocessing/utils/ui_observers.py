"""
File: smartcash/ui/dataset/preprocessing/utils/ui_observers.py
Deskripsi: Utilitas untuk observer UI pada modul preprocessing dataset
"""

from typing import Dict, Any, Callable, Optional, List
from smartcash.ui.utils.constants import ICONS
from enum import Enum, auto

# Event types untuk preprocessing
class PreprocessingEvents:
    """Event types untuk modul preprocessing."""
    PROGRESS_UPDATE = "preprocessing.progress_update"
    STATUS_UPDATE = "preprocessing.status_update"
    LOG_MESSAGE = "preprocessing.log_message"
    TASK_COMPLETED = "preprocessing.task_completed"
    PROCESS_START = "preprocessing.process_start"
    PROCESS_COMPLETE = "preprocessing.process_complete"
    PROCESS_ERROR = "preprocessing.process_error"
    PROCESS_STOP = "preprocessing.process_stop"

# Fallback ObserverManager untuk Colab environment
class MockObserverManager:
    """Mock Observer Manager untuk environment tanpa modul observer."""
    
    def __init__(self):
        """Inisialisasi mock observer manager."""
        self.observers = {}
        self.flags = {}
    
    def register(self, event_type: str, callback: Callable, group: str = None) -> None:
        """
        Register observer untuk event type.
        
        Args:
            event_type: Tipe event yang akan diobservasi
            callback: Fungsi callback untuk event
            group: Grup observer (opsional)
        """
        if event_type not in self.observers:
            self.observers[event_type] = []
        
        self.observers[event_type].append({
            'callback': callback,
            'group': group
        })
    
    def notify(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """
        Notifikasi semua observer untuk event type.
        
        Args:
            event_type: Tipe event yang dinotifikasi
            data: Data untuk dikirim ke observer
        """
        if event_type not in self.observers:
            return
        
        for observer in self.observers[event_type]:
            try:
                observer['callback'](data or {})
            except Exception as e:
                print(f"Error pada observer: {str(e)}")
    
    def set_flag(self, flag_name: str, value: Any) -> None:
        """
        Set flag value.
        
        Args:
            flag_name: Nama flag
            value: Nilai flag
        """
        self.flags[flag_name] = value
    
    def get_flag(self, flag_name: str, default: Any = None) -> Any:
        """
        Get flag value.
        
        Args:
            flag_name: Nama flag
            default: Nilai default jika flag tidak ada
            
        Returns:
            Nilai flag
        """
        return self.flags.get(flag_name, default)
        
    def cleanup(self) -> None:
        """
        Cleanup resources.
        """
        self.observers.clear()
        self.flags.clear()
        
    def is_running(self, process_name: str) -> bool:
        """
        Cek apakah proses sedang berjalan.
        
        Args:
            process_name: Nama proses
            
        Returns:
            True jika proses sedang berjalan, False jika tidak
        """
        return self.flags.get(f"{process_name}_running", False)
        
    def stop_all(self) -> None:
        """
        Stop semua proses.
        """
        self.set_flag('stop_requested', True)

def register_ui_observers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Register observer untuk UI preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Coba import observer dari common
    try:
        from smartcash.components.observer import ObserverManager, EventTopics
        observer_manager_class = ObserverManager
    except ImportError:
        try:
            from smartcash.common.observer import ObserverManager, EventType
            observer_manager_class = ObserverManager
        except ImportError:
            # Fallback ke mock observer manager
            from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
            log_message(ui_components, "Observer manager tidak tersedia. Menggunakan mock observer.", "warning", "⚠️")
            observer_manager_class = MockObserverManager
    
    # Cek apakah observer manager sudah ada
    if 'observer_manager' not in ui_components:
        ui_components['observer_manager'] = observer_manager_class()
    
    # Get observer manager
    observer_manager = ui_components['observer_manager']
    
    try:
        # Register event handlers
        register_progress_handlers(ui_components, observer_manager)
        register_status_handlers(ui_components, observer_manager)
        register_message_handlers(ui_components, observer_manager)
        register_completion_handlers(ui_components, observer_manager)
        
        # Setup observer group untuk UI
        ui_components['observer_group'] = "preprocessing_ui"
    except Exception as e:
        # Log warning jika terjadi error saat registrasi observer
        from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
        log_message(ui_components, f"Error saat registrasi observer: {str(e)}", "warning", "⚠️")
    
    return ui_components

def register_progress_handlers(ui_components: Dict[str, Any], observer_manager: Any) -> None:
    """
    Register handler untuk event progress.
    
    Args:
        ui_components: Dictionary komponen UI
        observer_manager: Observer manager
    """
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
    try:
        try:
            from smartcash.components.observer import EventTopics
            event_type = EventTopics.PROGRESS_UPDATE
        except ImportError:
            try:
                from smartcash.common.observer import EventType
                event_type = EventType.PROGRESS_UPDATE
            except ImportError:
                event_type = PreprocessingEvents.PROGRESS_UPDATE
        
        observer_manager.register(
            event_type=event_type,
            callback=handle_progress_update,
            group="preprocessing_ui"
        )
    except Exception:
        pass

def register_status_handlers(ui_components: Dict[str, Any], observer_manager: Any) -> None:
    """
    Register handler untuk event status.
    
    Args:
        ui_components: Dictionary komponen UI
        observer_manager: Observer manager
    """
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
    try:
        try:
            from smartcash.components.observer import EventTopics
            event_type = EventTopics.STATUS_UPDATE
        except ImportError:
            try:
                from smartcash.common.observer import EventType
                event_type = EventType.STATUS_UPDATE
            except ImportError:
                event_type = PreprocessingEvents.STATUS_UPDATE
        
        observer_manager.register(
            event_type=event_type,
            callback=handle_status_update,
            group="preprocessing_ui"
        )
    except Exception:
        pass

def register_message_handlers(ui_components: Dict[str, Any], observer_manager: Any) -> None:
    """
    Register handler untuk event message/log.
    
    Args:
        ui_components: Dictionary komponen UI
        observer_manager: Observer manager
    """
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
    try:
        try:
            from smartcash.components.observer import EventTopics
            event_type = EventTopics.LOG_MESSAGE
        except ImportError:
            try:
                from smartcash.common.observer import EventType
                event_type = EventType.LOG_MESSAGE
            except ImportError:
                event_type = PreprocessingEvents.LOG_MESSAGE
        
        observer_manager.register(
            event_type=event_type,
            callback=handle_log_message,
            group="preprocessing_ui"
        )
    except Exception:
        pass

def register_completion_handlers(ui_components: Dict[str, Any], observer_manager: Any) -> None:
    """
    Register handler untuk event completion.
    
    Args:
        ui_components: Dictionary komponen UI
        observer_manager: Observer manager
    """
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
    try:
        try:
            from smartcash.components.observer import EventTopics
            event_type = EventTopics.TASK_COMPLETED
        except ImportError:
            try:
                from smartcash.common.observer import EventType
                event_type = EventType.TASK_COMPLETED
            except ImportError:
                event_type = PreprocessingEvents.TASK_COMPLETED
        
        observer_manager.register(
            event_type=event_type,
            callback=handle_completion,
            group="preprocessing_ui"
        )
    except Exception:
        pass

def notify_process_start(ui_components: Dict[str, Any], process_name: str, display_info: str, split: Optional[str] = None) -> None:
    """
    Notifikasi observer bahwa proses telah dimulai.
    
    Args:
        ui_components: Dictionary komponen UI
        process_name: Nama proses yang dimulai
        display_info: Informasi tambahan untuk ditampilkan
        split: Split dataset yang diproses (opsional)
    """
    # Log informasi
    from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
    log_message(ui_components, f"Memulai {process_name} {display_info}", "info", ICONS['start'])
    
    # Notifikasi observer jika tersedia
    observer_manager = ui_components.get('observer_manager')
    if observer_manager and hasattr(observer_manager, 'notify'):
        try:
            try:
                from smartcash.components.observer import EventTopics
                event_type = EventTopics.PROCESS_START
            except ImportError:
                try:
                    from smartcash.common.observer import EventType
                    event_type = EventType.PROCESS_START
                except ImportError:
                    event_type = PreprocessingEvents.PROCESS_START
            
            observer_manager.notify(event_type, {
                'process_name': process_name,
                'display_info': display_info,
                'split': split
            })
        except Exception:
            pass
    
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
    # Log informasi
    from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
    log_message(ui_components, f"Preprocessing {display_info} selesai", "success", ICONS['success'])
    
    # Notifikasi observer jika tersedia
    observer_manager = ui_components.get('observer_manager')
    if observer_manager and hasattr(observer_manager, 'notify'):
        try:
            try:
                from smartcash.components.observer import EventTopics
                event_type = EventTopics.PROCESS_COMPLETE
            except ImportError:
                try:
                    from smartcash.common.observer import EventType
                    event_type = EventType.PROCESS_COMPLETE
                except ImportError:
                    event_type = PreprocessingEvents.PROCESS_COMPLETE
            
            observer_manager.notify(event_type, {
                'success': True,
                'result': result,
                'display_info': display_info
            })
        except Exception:
            pass
    
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
    # Log informasi
    from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
    log_message(ui_components, f"Error pada preprocessing: {error_message}", "error", ICONS['error'])
    
    # Notifikasi observer jika tersedia
    observer_manager = ui_components.get('observer_manager')
    if observer_manager and hasattr(observer_manager, 'notify'):
        try:
            try:
                from smartcash.components.observer import EventTopics
                event_type = EventTopics.PROCESS_ERROR
            except ImportError:
                try:
                    from smartcash.common.observer import EventType
                    event_type = EventType.PROCESS_ERROR
                except ImportError:
                    event_type = PreprocessingEvents.PROCESS_ERROR
            
            observer_manager.notify(event_type, {
                'success': False,
                'error': error_message
            })
        except Exception:
            pass
    
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
    # Log informasi
    from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
    log_message(ui_components, f"Proses preprocessing dihentikan oleh pengguna", "warning", ICONS['stop'])
    
    # Notifikasi observer jika tersedia
    observer_manager = ui_components.get('observer_manager')
    if observer_manager and hasattr(observer_manager, 'notify'):
        try:
            try:
                from smartcash.components.observer import EventTopics
                event_type = EventTopics.PROCESS_STOP
            except ImportError:
                try:
                    from smartcash.common.observer import EventType
                    event_type = EventType.PROCESS_STOP
                except ImportError:
                    event_type = PreprocessingEvents.PROCESS_STOP
            
            observer_manager.notify(event_type, {
                'display_info': display_info
            })
        except Exception:
            pass
    
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
