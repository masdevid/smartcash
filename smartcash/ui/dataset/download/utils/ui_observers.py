"""
File: smartcash/ui/dataset/download/utils/ui_observers.py
Deskripsi: Observer untuk menangani notifikasi UI pada proses download dataset
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.components.observer import BaseObserver
from smartcash.ui.dataset.download.utils.notification_manager import DownloadUIEvents

class LogOutputObserver(BaseObserver):
    """Observer untuk menangani notifikasi log ke output UI"""
    
    def __init__(self, log_output: widgets.Output, priority: int = 0, name: str = None):
        """
        Inisialisasi observer log output
        
        Args:
            log_output: Widget output untuk menampilkan log
            priority: Prioritas observer
            name: Nama observer
        """
        self.log_output = log_output
        self.priority = priority
        self.name = name or self.__class__.__name__
        self.event_types = [
            DownloadUIEvents.LOG_INFO,
            DownloadUIEvents.LOG_WARNING,
            DownloadUIEvents.LOG_ERROR,
            DownloadUIEvents.LOG_SUCCESS
        ]
        
    def update(self, event_type: str, sender: Any, **kwargs):
        """
        Update log output berdasarkan notifikasi
        
        Args:
            event_type: Tipe event
            sender: Objek pengirim notifikasi
            **kwargs: Parameter tambahan
        """
        message = kwargs.get('message', '')
        level = kwargs.get('level', 'info')
        
        # Map level ke warna dan emoji
        color_map = {
            'info': 'blue',
            'warning': 'orange',
            'error': 'red',
            'success': 'green'
        }
        emoji_map = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'success': '✅'
        }
        
        color = color_map.get(level, 'black')
        emoji = emoji_map.get(level, '')
        
        # Tampilkan pesan di log output
        with self.log_output:
            print(f"<span style='color: {color}'>{emoji} {message}</span>")


class ProgressBarObserver(BaseObserver):
    """Observer untuk menangani notifikasi progress bar"""
    
    def __init__(self, ui_components: Dict[str, Any], priority: int = 0, name: str = None):
        """
        Inisialisasi observer progress bar
        
        Args:
            ui_components: Dictionary berisi komponen UI
            priority: Prioritas observer
            name: Nama observer
        """
        self.ui_components = ui_components
        self.priority = priority
        self.name = name or self.__class__.__name__
        self.event_types = [
            DownloadUIEvents.PROGRESS_START,
            DownloadUIEvents.PROGRESS_UPDATE,
            DownloadUIEvents.PROGRESS_COMPLETE,
            DownloadUIEvents.PROGRESS_ERROR
        ]
        
    def update(self, event_type: str, sender: Any, **kwargs):
        """
        Update progress bar berdasarkan notifikasi
        
        Args:
            event_type: Tipe event
            sender: Objek pengirim notifikasi
            **kwargs: Parameter tambahan
        """
        # Pastikan progress container ditampilkan
        self.ui_components['progress_container'].layout.display = 'block'
        
        # Ambil parameter dari kwargs
        progress = kwargs.get('progress', 0)
        total = kwargs.get('total', 100)
        percentage = kwargs.get('percentage', 0)
        message = kwargs.get('message', '')
        step = kwargs.get('step', 1)
        total_steps = kwargs.get('total_steps', 1)
        
        # Update progress bar
        if 'progress_bar' in self.ui_components:
            progress_bar = self.ui_components['progress_bar']
            if hasattr(progress_bar, 'value'):
                # Ensure percentage is an integer
                try:
                    percentage = int(float(percentage))
                except (ValueError, TypeError):
                    percentage = 0
                progress_bar.value = percentage
        
        # Update label progress
        if 'step_label' in self.ui_components and message:
            self.ui_components['step_label'].value = message
            
        # Update overall label jika ada
        if 'overall_label' in self.ui_components:
            if event_type == DownloadUIEvents.PROGRESS_START:
                self.ui_components['overall_label'].value = f"Memulai proses download..."
            elif event_type == DownloadUIEvents.PROGRESS_COMPLETE:
                self.ui_components['overall_label'].value = f"Download selesai"
            elif event_type == DownloadUIEvents.PROGRESS_ERROR:
                self.ui_components['overall_label'].value = f"Terjadi kesalahan dalam proses download"
            else:
                # Progress update
                if step and total_steps:
                    self.ui_components['overall_label'].value = f"Langkah {step}/{total_steps} ({percentage}%)"
                else:
                    self.ui_components['overall_label'].value = f"Progress: {percentage}%"
                    
        # Sembunyikan progress container jika selesai atau error
        if event_type in [DownloadUIEvents.PROGRESS_COMPLETE, DownloadUIEvents.PROGRESS_ERROR]:
            # Jangan langsung sembunyikan, beri waktu untuk melihat hasil akhir
            pass


def register_ui_observers(ui_components: Dict[str, Any], observer_manager=None):
    """
    Mendaftarkan observer UI untuk menangani notifikasi
    
    Args:
        ui_components: Dictionary berisi komponen UI
        observer_manager: Observer manager opsional
    """
    from smartcash.components.observer import get_observer_manager
    
    # Dapatkan observer manager jika tidak disediakan
    if observer_manager is None:
        observer_manager = get_observer_manager()
    
    # Daftarkan observer log output
    if 'log_output' in ui_components:
        log_observer = LogOutputObserver(ui_components['log_output'])
        observer_manager.register(log_observer)
    
    # Daftarkan observer progress bar
    progress_observer = ProgressBarObserver(ui_components)
    observer_manager.register(progress_observer)
    
    return observer_manager
