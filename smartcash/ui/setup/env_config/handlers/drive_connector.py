"""
File: smartcash/ui/setup/env_config/handlers/drive_connector.py
Deskripsi: Handler untuk koneksi Google Drive - SRP untuk drive functionality
"""

from typing import Dict, Any, Tuple
from pathlib import Path

from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.common.environment import get_environment_manager

class DriveConnector:
    """Handler untuk koneksi Google Drive - fokus hanya pada drive operations"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi connector dengan UI components
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self.logger = create_ui_logger_bridge(ui_components, "drive_connector")
        self.env_manager = get_environment_manager()
    
    def ensure_drive_mounted(self) -> Tuple[bool, str]:
        """
        Ensure Google Drive ter-mount di Colab
        
        Returns:
            Tuple (success, message)
        """
        # Skip jika bukan Colab
        if not self._is_colab():
            return True, "Local environment - no drive needed"
        
        try:
            # Check jika sudah ter-mount
            if self.env_manager.is_drive_mounted:
                self.logger.success("📱 Google Drive sudah terhubung")
                return True, "Drive already mounted"
            
            # Mount drive
            self.logger.info("📱 Menghubungkan Google Drive...")
            success, message = self.env_manager.mount_drive()
            
            if success:
                self.logger.success("✅ Google Drive berhasil terhubung")
                return True, message
            else:
                self.logger.error(f"❌ Gagal menghubungkan Drive: {message}")
                return False, message
                
        except Exception as e:
            error_msg = f"Error mounting drive: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return False, error_msg
    
    def get_drive_paths(self) -> Dict[str, Path]:
        """
        Get path direktori Drive
        
        Returns:
            Dictionary path Drive
        """
        if not self._is_colab() or not self.env_manager.is_drive_mounted:
            return {}
        
        drive_base = self.env_manager.drive_path
        return {
            'base': drive_base,
            'data': drive_base / 'data',
            'configs': drive_base / 'configs',
            'exports': drive_base / 'exports',
            'logs': drive_base / 'logs',
            'models': drive_base / 'models',
            'output': drive_base / 'output'
        }
    
    def create_drive_directories(self) -> bool:
        """
        Buat direktori yang diperlukan di Drive
        
        Returns:
            Success status
        """
        if not self._is_colab():
            return True
            
        try:
            paths = self.get_drive_paths()
            if not paths:
                return False
            
            created_count = 0
            for name, path in paths.items():
                if name == 'base':  # Skip base path
                    continue
                    
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"📁 Dibuat direktori Drive: {name}")
                    created_count += 1
            
            if created_count > 0:
                self.logger.success(f"✅ Dibuat {created_count} direktori di Drive")
            else:
                self.logger.info("📁 Semua direktori Drive sudah ada")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error creating drive directories: {str(e)}")
            return False
    
    def _is_colab(self) -> bool:
        """Check apakah running di Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False