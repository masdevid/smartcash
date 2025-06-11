"""
File: smartcash/ui/utils/notification_manager.py
Deskripsi: Manager untuk mengelola notifikasi ke UI
"""

from typing import Dict, Any, Optional
import time

from smartcash.common.logger import get_logger


class NotificationManager:
    """Manager untuk mengelola notifikasi ke UI"""
    
    def __init__(self, ui_components: Dict[str, Any], logger=None):
        """Inisialisasi notification manager
        
        Args:
            ui_components: Komponen UI untuk notifikasi
            logger: Logger untuk logging
        """
        self.ui_components = ui_components
        self.logger = logger or get_logger(__name__)
        self._last_update_time = 0
        self._min_update_interval = 0.1  # 100ms minimum interval
    
    def notify_info(self, message: str) -> None:
        """Notifikasi informasi
        
        Args:
            message: Pesan informasi
        """
        self._update_status(message, 'info')
    
    def notify_success(self, message: str) -> None:
        """Notifikasi sukses
        
        Args:
            message: Pesan sukses
        """
        self._update_status(message, 'success')
    
    def notify_warning(self, message: str) -> None:
        """Notifikasi peringatan
        
        Args:
            message: Pesan peringatan
        """
        self._update_status(message, 'warning')
    
    def notify_error(self, message: str) -> None:
        """Notifikasi error
        
        Args:
            message: Pesan error
        """
        self._update_status(message, 'error')
    
    def notify_process_start(self, message: str) -> None:
        """Notifikasi awal proses
        
        Args:
            message: Pesan awal proses
        """
        self._update_status(message, 'info')
    
    def notify_process_complete(self, message: str) -> None:
        """Notifikasi akhir proses
        
        Args:
            message: Pesan akhir proses
        """
        self._update_status(message, 'success')
    
    def notify_process_error(self, message: str) -> None:
        """Notifikasi error proses
        
        Args:
            message: Pesan error proses
        """
        self._update_status(message, 'error')
    
    def update_progress(self, current: int, total: int, message: str) -> None:
        """Update progress bar
        
        Args:
            current: Nilai progress saat ini
            total: Nilai progress total
            message: Pesan progress
        """
        try:
            # Cek apakah sudah waktunya update
            current_time = time.time()
            if current_time - self._last_update_time < self._min_update_interval:
                return
            
            # Update progress bar
            progress_bar = self.ui_components.get('progress_bar')
            if progress_bar and hasattr(progress_bar, 'value') and hasattr(progress_bar, 'max'):
                progress_bar.value = current
                progress_bar.max = total
                
                # Update progress label
                progress_label = self.ui_components.get('progress_label')
                if progress_label and hasattr(progress_label, 'value'):
                    progress_label.value = f"{message} ({current}/{total})"
            
            # Update status
            self._update_status(message, 'info')
            
            # Update last update time
            self._last_update_time = current_time
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat update progress: {str(e)}")
    
    def update_status(self, message: str, status_type: str = 'info') -> None:
        """Update status panel
        
        Args:
            message: Pesan status
            status_type: Tipe status (info, success, warning, error)
        """
        self._update_status(message, status_type)
    
    def _update_status(self, message: str, status_type: str) -> None:
        """Update status panel
        
        Args:
            message: Pesan status
            status_type: Tipe status (info, success, warning, error)
        """
        try:
            # Cek apakah sudah waktunya update
            current_time = time.time()
            if current_time - self._last_update_time < self._min_update_interval:
                return
            
            # Update status panel
            status_panel = self.ui_components.get('status_panel')
            if status_panel and hasattr(status_panel, 'value'):
                # Tambahkan emoji sesuai tipe status
                emoji = self._get_status_emoji(status_type)
                status_panel.value = f"{emoji} {message}"
                
                # Set style sesuai tipe status
                if hasattr(status_panel, 'style'):
                    status_panel.style = self._get_status_style(status_type)
            
            # Update last update time
            self._last_update_time = current_time
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat update status: {str(e)}")
    
    def _get_status_emoji(self, status_type: str) -> str:
        """Dapatkan emoji sesuai tipe status
        
        Args:
            status_type: Tipe status (info, success, warning, error)
            
        Returns:
            str: Emoji status
        """
        if status_type == 'success':
            return "✅"
        elif status_type == 'warning':
            return "⚠️"
        elif status_type == 'error':
            return "❌"
        else:
            return "ℹ️"
    
    def _get_status_style(self, status_type: str) -> Dict[str, str]:
        """Dapatkan style sesuai tipe status
        
        Args:
            status_type: Tipe status (info, success, warning, error)
            
        Returns:
            Dict[str, str]: Style status
        """
        if status_type == 'success':
            return {'color': 'green'}
        elif status_type == 'warning':
            return {'color': 'orange'}
        elif status_type == 'error':
            return {'color': 'red'}
        else:
            return {'color': 'blue'}
