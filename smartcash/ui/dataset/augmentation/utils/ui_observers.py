"""
File: smartcash/ui/dataset/augmentation/utils/ui_observers.py
Deskripsi: Utility untuk observer pattern pada modul augmentasi dataset (diperbaiki args)
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import ICONS

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
    if logger: logger.info(f"{ICONS['start']} Memulai {process_name} {display_info}")
    
    # Panggil callback jika tersedia
    if 'on_process_start' in ui_components and callable(ui_components['on_process_start']):
        ui_components['on_process_start']("augmentation", {
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
    logger = ui_components.get('logger')
    if logger: logger.info(f"{ICONS['success']} Augmentasi {display_info} selesai")
    
    # Panggil callback jika tersedia
    if 'on_process_complete' in ui_components and callable(ui_components['on_process_complete']):
        ui_components['on_process_complete']("augmentation", result)

def notify_process_error(ui_components: Dict[str, Any], error_message: str) -> None:
    """
    Notifikasi observer bahwa proses mengalami error.
    
    Args:
        ui_components: Dictionary komponen UI
        error_message: Pesan error yang terjadi
    """
    logger = ui_components.get('logger')
    if logger: logger.error(f"{ICONS['error']} Error pada augmentasi: {error_message}")
    
    # Panggil callback jika tersedia
    if 'on_process_error' in ui_components and callable(ui_components['on_process_error']):
        ui_components['on_process_error']("augmentation", error_message)

def notify_process_stop(ui_components: Dict[str, Any]) -> None:
    """
    Notifikasi observer bahwa proses telah dihentikan oleh pengguna.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger')
    if logger: logger.warning(f"{ICONS['warning']} Augmentasi dihentikan oleh pengguna")
    
    # Panggil callback jika tersedia
    if 'on_process_stop' in ui_components and callable(ui_components['on_process_stop']):
        ui_components['on_process_stop']("augmentation")

def disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True) -> None:
    """
    Nonaktifkan UI selama proses berjalan.
    
    Args:
        ui_components: Dictionary komponen UI
        disable: True untuk nonaktifkan, False untuk aktifkan
    """
    # Daftar tombol yang akan dinonaktifkan
    button_keys = ['augment_button', 'reset_button', 'save_button', 'cleanup_button']
    
    # Daftar komponen lain yang akan dinonaktifkan
    component_keys = ['split_selector', 'augmentation_options', 'advanced_options']
    
    # Nonaktifkan tombol
    for key in button_keys:
        if key in ui_components and hasattr(ui_components[key], 'disabled'):
            ui_components[key].disabled = disable
    
    # Nonaktifkan komponen lain
    for key in component_keys:
        if key in ui_components and hasattr(ui_components[key], 'disabled'):
            ui_components[key].disabled = disable
    
    # Tampilkan tombol stop jika proses sedang berjalan, sembunyikan jika tidak
    if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
        ui_components['stop_button'].layout.display = 'block' if disable else 'none'
    
    # Tampilkan tombol augment jika proses tidak sedang berjalan, sembunyikan jika sedang berjalan
    if 'augment_button' in ui_components and hasattr(ui_components['augment_button'], 'layout'):
        ui_components['augment_button'].layout.display = 'none' if disable else 'block'
        
def register_ui_observers(ui_components: Dict[str, Any]) -> Any:
    """
    Register UI observers untuk notifikasi proses augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Observer manager instance
    """
    try:
        # Coba import observer manager
        from smartcash.components.observer import ObserverManager
        
        # Buat observer manager jika belum ada
        if 'observer_manager' not in ui_components:
            observer_manager = ObserverManager()
            ui_components['observer_manager'] = observer_manager
        else:
            observer_manager = ui_components['observer_manager']
        
        # Setup observer group untuk augmentasi
        observer_group = 'augmentation_observers'
        ui_components['observer_group'] = observer_group
        
        # Register observer callbacks dengan format yang benar (2-3 args)
        def on_process_start(event_type: str, data: Dict[str, Any] = None):
            if data:
                notify_process_start(ui_components, event_type, data.get('display_info', ''), data.get('split'))
        
        def on_process_complete(event_type: str, data: Dict[str, Any] = None):
            if data:
                notify_process_complete(ui_components, data, data.get('display_info', ''))
        
        def on_process_error(event_type: str, data: str = None):
            if data:
                notify_process_error(ui_components, data)
        
        def on_process_stop(event_type: str, data: Any = None):
            notify_process_stop(ui_components)
        
        # Register callbacks ke observer manager dengan format yang benar
        observer_manager.register('process_start', on_process_start)
        observer_manager.register('process_complete', on_process_complete)
        observer_manager.register('process_error', on_process_error)
        observer_manager.register('process_stop', on_process_stop)
        
        return observer_manager
        
    except ImportError:
        # Fallback jika observer manager tidak tersedia
        from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message
        log_message(ui_components, "Observer manager tidak tersedia, menggunakan fallback", "warning", "⚠️")
        
        # Buat mock observer manager
        ui_components['observer_manager'] = MockObserverManager()
        return ui_components['observer_manager']

class MockObserverManager:
    """Mock observer manager untuk fallback."""
    
    def register(self, event: str, callback):
        """Mock register method dengan 2 args."""
        pass
    
    def notify(self, event: str, data: Any = None):
        """Mock notify method."""
        pass
    
    def unregister_group(self, group: str):
        """Mock unregister method."""
        pass