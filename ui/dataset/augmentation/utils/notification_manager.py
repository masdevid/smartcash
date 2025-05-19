"""
File: smartcash/ui/dataset/augmentation/utils/notification_manager.py
Deskripsi: Manager untuk notifikasi pada modul augmentasi dataset
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.common.logger import get_logger

class NotificationManager:
    """Manager untuk notifikasi pada modul augmentasi dataset."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi NotificationManager.
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self.logger = ui_components.get('logger', get_logger('augmentation'))
    
    def update_status(self, status_type: str, message: str) -> None:
        """
        Update status panel dengan pesan dan tipe yang ditentukan.
        
        Args:
            status_type: Tipe pesan ('info', 'success', 'warning', 'error')
            message: Pesan yang akan ditampilkan
        """
        if 'update_status_panel' in self.ui_components and callable(self.ui_components['update_status_panel']):
            self.ui_components['update_status_panel'](self.ui_components, status_type, message)
        else:
            # Fallback jika update_status_panel tidak tersedia
            if status_type == 'error':
                self.logger.error(message)
            elif status_type == 'warning':
                self.logger.warning(message)
            elif status_type == 'success':
                self.logger.info(message)
            else:
                self.logger.info(message)
    
    def update_progress(self, progress: float, message: str = "") -> None:
        """
        Update progress bar dengan nilai dan pesan yang ditentukan.
        
        Args:
            progress: Nilai progress (0-100)
            message: Pesan yang akan ditampilkan
        """
        if 'progress_callback' in self.ui_components and callable(self.ui_components['progress_callback']):
            self.ui_components['progress_callback'](progress, message)
    
    def notify_process_start(self, process_name: str, display_info: str, split: Optional[str] = None) -> None:
        """
        Notifikasi bahwa proses telah dimulai.
        
        Args:
            process_name: Nama proses yang dimulai
            display_info: Informasi tambahan untuk ditampilkan
            split: Split dataset yang diproses (opsional)
        """
        # Update status
        self.update_status('info', f"{ICONS['start']} Memulai {process_name} {display_info}")
        
        # Panggil fungsi notifikasi jika tersedia
        if 'notify_process_start' in self.ui_components and callable(self.ui_components['notify_process_start']):
            self.ui_components['notify_process_start'](self.ui_components, process_name, display_info, split)
    
    def notify_process_complete(self, result: Dict[str, Any], display_info: str) -> None:
        """
        Notifikasi bahwa proses telah selesai dengan sukses.
        
        Args:
            result: Dictionary hasil proses
            display_info: Informasi tambahan untuk ditampilkan
        """
        # Update status
        self.update_status('success', f"{ICONS['success']} Augmentasi {display_info} selesai")
        
        # Panggil fungsi notifikasi jika tersedia
        if 'notify_process_complete' in self.ui_components and callable(self.ui_components['notify_process_complete']):
            self.ui_components['notify_process_complete'](self.ui_components, result, display_info)
    
    def notify_process_error(self, error_message: str) -> None:
        """
        Notifikasi bahwa proses mengalami error.
        
        Args:
            error_message: Pesan error yang terjadi
        """
        # Update status
        self.update_status('error', f"{ICONS['error']} Error pada augmentasi: {error_message}")
        
        # Panggil fungsi notifikasi jika tersedia
        if 'notify_process_error' in self.ui_components and callable(self.ui_components['notify_process_error']):
            self.ui_components['notify_process_error'](self.ui_components, error_message)

def get_notification_manager(ui_components: Dict[str, Any]) -> NotificationManager:
    """
    Dapatkan instance NotificationManager.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Instance NotificationManager
    """
    # Cek apakah notification_manager sudah ada di ui_components
    if 'notification_manager' in ui_components:
        return ui_components['notification_manager']
    
    # Buat instance baru
    notification_manager = NotificationManager(ui_components)
    ui_components['notification_manager'] = notification_manager
    
    return notification_manager
