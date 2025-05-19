"""
File: smartcash/ui/dataset/preprocessing/utils/notification_manager.py
Deskripsi: Utilitas untuk mengelola notifikasi preprocessing dataset
"""

from typing import Dict, Any, Optional, List, Callable
from IPython.display import display
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

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
