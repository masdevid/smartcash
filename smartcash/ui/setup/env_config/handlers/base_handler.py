"""
File: smartcash/ui/setup/env_config/handlers/base_handler.py
Deskripsi: Handler dasar yang berisi fungsionalitas umum untuk semua handler env_config - dikurangi duplikasi dan diperbaiki logging
"""

from typing import Dict, Any, Optional, Callable
import logging

from smartcash.ui.utils.ui_logger import create_ui_logger
from smartcash.ui.utils.ui_logger_namespace import ENV_CONFIG_LOGGER_NAMESPACE

class BaseHandler:
    """
    Handler dasar dengan fungsionalitas umum untuk semua handler environment config
    Dikurangi duplikasi dan diperbaiki logging management
    """
    
    def __init__(self, ui_components: Dict[str, Any] = None, 
                ui_callback: Optional[Dict[str, Callable]] = None):
        """
        Inisialisasi handler dasar dengan UI components atau callbacks
        
        Args:
            ui_components: Dictionary komponen UI (preferred method)
            ui_callback: Dictionary callback untuk update UI (legacy support)
        """
        self.ui_components = ui_components or {}
        self.ui_callback = ui_callback or {}
        
        # Setup logger dengan UI integration jika UI components tersedia
        if self.ui_components:
            # Setup namespace untuk environment config
            if 'logger_namespace' not in self.ui_components:
                self.ui_components['logger_namespace'] = ENV_CONFIG_LOGGER_NAMESPACE
            if 'env_config_initialized' not in self.ui_components:
                self.ui_components['env_config_initialized'] = True
                
            # Buat UI logger yang terintegrasi
            self.logger = create_ui_logger(
                self.ui_components, 
                ENV_CONFIG_LOGGER_NAMESPACE,
                redirect_stdout=False  # Prevent stdout hijacking
            )
        else:
            # Fallback ke logger biasa jika tidak ada UI components
            from smartcash.common.logger import get_logger
            self.logger = get_logger(ENV_CONFIG_LOGGER_NAMESPACE)
            
        # Suppress verbose logging dari dependencies
        self._suppress_dependency_logs()
    
    def _suppress_dependency_logs(self):
        """Suppress log yang tidak perlu dari dependencies"""
        # Suppress environment manager logs yang verbose
        for logger_name in [
            'smartcash.common.environment',
            'smartcash.common.config.manager',
            'google.colab',
            'urllib3'
        ]:
            dep_logger = logging.getLogger(logger_name)
            dep_logger.setLevel(logging.ERROR)  # Only show errors
    
    def _log_message(self, message: str, level: str = "info", icon: str = None):
        """
        Log message ke logger dan UI (consolidated method)
        
        Args:
            message: Pesan yang akan dilog
            level: Level log (info, success, warning, error)
            icon: Ikon opsional untuk ditampilkan di UI
        """
        # Log ke logger object berdasarkan level
        if level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "success":
            self.logger.success(message)
        else:
            self.logger.info(message)
            
        # Gunakan callback jika ada (legacy support)
        if 'log_message' in self.ui_callback:
            self.ui_callback['log_message'](message, level, icon)
    
    def _update_status(self, message: str, status_type: str = "info"):
        """Update status UI jika callback tersedia"""
        if 'update_status' in self.ui_callback:
            self.ui_callback['update_status'](message, status_type)
        elif 'status_panel' in self.ui_components:
            # Direct update ke status panel jika tersedia
            from smartcash.ui.components.status_panel import update_status_panel
            update_status_panel(self.ui_components['status_panel'], message, status_type)
    
    def _update_progress(self, value: float, message: str = ""):
        """Update progress UI jika callback tersedia"""
        if 'update_progress' in self.ui_callback:
            self.ui_callback['update_progress'](value, message)
        elif 'progress_bar' in self.ui_components:
            # Direct update ke progress bar jika tersedia
            from smartcash.ui.components.progress_tracking import update_progress
            update_progress(
                self.ui_components, 
                int(value * 100), 
                100, 
                message
            )
    
    def _get_config_manager(self):
        """Get config manager instance dengan error handling"""
        try:
            from smartcash.common.config import get_config_manager
            return get_config_manager()
        except Exception as e:
            self._log_message(f"Error getting config manager: {str(e)}", "error", "❌")
            return None
    
    def _get_environment_manager(self):
        """Get environment manager instance dengan error handling"""
        try:
            from smartcash.common.environment import get_environment_manager
            return get_environment_manager()
        except Exception as e:
            self._log_message(f"Error getting environment manager: {str(e)}", "error", "❌")
            return None


class EnvConfigHandlerMixin:
    """
    Mixin untuk handler environment config yang menyediakan method umum
    """
    
    def check_colab_environment(self) -> bool:
        """Check apakah berjalan di Google Colab"""
        try:
            from smartcash.common.utils import is_colab
            return is_colab()
        except ImportError:
            # Fallback detection
            try:
                import google.colab
                return True
            except ImportError:
                return False
    
    def ensure_drive_connection(self) -> bool:
        """Ensure Google Drive terhubung di Colab"""
        if not self.check_colab_environment():
            return True  # Not in Colab, no need to mount
            
        try:
            env_manager = self._get_environment_manager()
            if env_manager and not env_manager.is_drive_mounted:
                success, message = env_manager.mount_drive()
                if success:
                    self._log_message("Google Drive berhasil terhubung", "success", "✅")
                else:
                    self._log_message(f"Gagal menghubungkan Google Drive: {message}", "error", "❌")
                return success
            return True
        except Exception as e:
            self._log_message(f"Error saat koneksi Drive: {str(e)}", "error", "❌")
            return False
    
    def get_required_directories(self) -> list:
        """Get list direktori yang diperlukan"""
        return [
            'smartcash', 'yolov5', 'data', 'exports', 
            'logs', 'models', 'output', 'configs'
        ]
    
    def validate_environment_setup(self) -> Dict[str, Any]:
        """Validate setup environment dan return status"""
        status = {
            'is_colab': self.check_colab_environment(),
            'drive_connected': False,
            'directories_exist': {},
            'config_files_exist': {},
            'ready': False
        }
        
        try:
            # Check drive connection
            env_manager = self._get_environment_manager()
            if env_manager:
                status['drive_connected'] = env_manager.is_drive_mounted
                status['base_dir'] = str(env_manager.base_dir)
            
            # Check directories
            if status['is_colab']:
                from pathlib import Path
                for dir_name in self.get_required_directories():
                    dir_path = Path(f"/content/{dir_name}")
                    status['directories_exist'][dir_name] = dir_path.exists()
            
            # Check if ready
            status['ready'] = (
                not status['is_colab'] or  # Not in Colab, or
                (status['drive_connected'] and  # Drive connected and
                 all(status['directories_exist'].values()))  # All dirs exist
            )
            
        except Exception as e:
            self._log_message(f"Error validating environment: {str(e)}", "error", "❌")
            
        return status