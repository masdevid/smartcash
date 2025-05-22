"""
File: smartcash/ui/dataset/augmentation/utils/ui_observers.py
Deskripsi: Utility untuk observer pattern dengan BaseObserver compliance dan logger bridge
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

# Import BaseObserver untuk compliance
try:
    from smartcash.components.observer.base_observer import BaseObserver
    from smartcash.components.observer.manager_observer import get_observer_manager
    OBSERVER_AVAILABLE = True
except ImportError:
    OBSERVER_AVAILABLE = False
    BaseObserver = None

class UIAugmentationObserver(BaseObserver if OBSERVER_AVAILABLE else object):
    """Observer khusus untuk UI augmentasi dataset yang comply dengan BaseObserver."""
    
    def __init__(self, ui_components: Dict[str, Any], name: str = "ui_augmentation"):
        """
        Inisialisasi UI Augmentation Observer.
        
        Args:
            ui_components: Dictionary komponen UI
            name: Nama observer
        """
        self.ui_components = ui_components
        self.ui_logger = create_ui_logger_bridge(ui_components, "ui_augmentation_observer")
        
        # Inisialisasi BaseObserver jika tersedia
        if OBSERVER_AVAILABLE and BaseObserver:
            super().__init__(
                name=name,
                priority=50,
                event_filter=['process_start', 'process_complete', 'process_error', 'process_stop']
            )
        else:
            self.name = name
            self.priority = 50
    
    def update(self, event_type: str, sender: Any, **kwargs) -> None:
        """
        Handle event updates dari observer manager.
        
        Args:
            event_type: Tipe event yang diterima
            sender: Pengirim event
            **kwargs: Data tambahan event
        """
        try:
            if event_type == 'process_start':
                self._handle_process_start(kwargs)
            elif event_type == 'process_complete':
                self._handle_process_complete(kwargs)
            elif event_type == 'process_error':
                self._handle_process_error(kwargs)
            elif event_type == 'process_stop':
                self._handle_process_stop(kwargs)
        except Exception as e:
            self.ui_logger.warning(f"⚠️ Error handling {event_type}: {str(e)}")
    
    def _handle_process_start(self, data: Dict[str, Any]) -> None:
        """Handle process start event."""
        process_name = data.get('process_name', 'augmentation')
        display_info = data.get('display_info', '')
        split = data.get('split')
        
        notify_process_start(self.ui_components, process_name, display_info, split)
    
    def _handle_process_complete(self, data: Dict[str, Any]) -> None:
        """Handle process complete event."""
        result = data.get('result', {})
        display_info = data.get('display_info', '')
        
        notify_process_complete(self.ui_components, result, display_info)
    
    def _handle_process_error(self, data: Dict[str, Any]) -> None:
        """Handle process error event."""
        error_message = data.get('error_message', 'Unknown error')
        
        notify_process_error(self.ui_components, error_message)
    
    def _handle_process_stop(self, data: Dict[str, Any]) -> None:
        """Handle process stop event."""
        notify_process_stop(self.ui_components)

def notify_process_start(ui_components: Dict[str, Any], process_name: str, display_info: str, split: Optional[str] = None) -> None:
    """
    Notifikasi observer bahwa proses telah dimulai.
    
    Args:
        ui_components: Dictionary komponen UI
        process_name: Nama proses yang dimulai
        display_info: Informasi tambahan untuk ditampilkan
        split: Split dataset yang diproses (opsional)
    """
    # Setup logger bridge jika belum ada
    if 'logger' not in ui_components:
        ui_components['logger'] = create_ui_logger_bridge(ui_components, "ui_observers")
    
    ui_components['logger'].info(f"{ICONS['start']} Memulai {process_name} {display_info}")
    
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
    # Setup logger bridge jika belum ada
    if 'logger' not in ui_components:
        ui_components['logger'] = create_ui_logger_bridge(ui_components, "ui_observers")
    
    ui_components['logger'].success(f"{ICONS['success']} Augmentasi {display_info} selesai")
    
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
    # Setup logger bridge jika belum ada
    if 'logger' not in ui_components:
        ui_components['logger'] = create_ui_logger_bridge(ui_components, "ui_observers")
    
    ui_components['logger'].error(f"{ICONS['error']} Error pada augmentasi: {error_message}")
    
    # Panggil callback jika tersedia
    if 'on_process_error' in ui_components and callable(ui_components['on_process_error']):
        ui_components['on_process_error']("augmentation", error_message)

def notify_process_stop(ui_components: Dict[str, Any]) -> None:
    """
    Notifikasi observer bahwa proses telah dihentikan oleh pengguna.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Setup logger bridge jika belum ada
    if 'logger' not in ui_components:
        ui_components['logger'] = create_ui_logger_bridge(ui_components, "ui_observers")
    
    ui_components['logger'].warning(f"{ICONS['warning']} Augmentasi dihentikan oleh pengguna")
    
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
    Register UI observers untuk notifikasi proses augmentasi dengan BaseObserver compliance.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Observer manager instance
    """
    # Setup logger bridge untuk error reporting
    if 'logger' not in ui_components:
        ui_components['logger'] = create_ui_logger_bridge(ui_components, "ui_observers")
    
    try:
        if not OBSERVER_AVAILABLE:
            ui_components['logger'].warning("⚠️ Observer system tidak tersedia, menggunakan fallback")
            ui_components['observer_manager'] = MockObserverManager()
            return ui_components['observer_manager']
        
        # Dapatkan observer manager
        observer_manager = get_observer_manager()
        ui_components['observer_manager'] = observer_manager
        
        # Setup observer group untuk augmentasi
        observer_group = 'augmentation_observers'
        ui_components['observer_group'] = observer_group
        
        # Buat UI observer yang comply dengan BaseObserver
        ui_observer = UIAugmentationObserver(ui_components, "augmentation_ui_observer")
        
        # Event types yang akan diobservasi
        event_types = ['process_start', 'process_complete', 'process_error', 'process_stop']
        
        # Register observer dengan event types
        for event_type in event_types:
            observer_manager.register(ui_observer, [event_type])
        
        ui_components['logger'].success("✅ UI observers berhasil didaftarkan")
        return observer_manager
        
    except Exception as e:
        # Fallback jika observer manager gagal
        ui_components['logger'].warning(f"⚠️ Observer manager gagal setup: {str(e)}, menggunakan fallback")
        ui_components['observer_manager'] = MockObserverManager()
        return ui_components['observer_manager']

class MockObserverManager:
    """Mock observer manager untuk fallback saat observer system tidak tersedia."""
    
    def register(self, observer, event_types=None):
        """Mock register method."""
        pass
    
    def notify(self, event: str, data: Any = None):
        """Mock notify method."""
        pass
    
    def unregister_group(self, group: str):
        """Mock unregister method."""
        pass
    
    def unregister_all(self):
        """Mock unregister all method."""
        pass