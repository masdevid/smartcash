"""
File: smartcash/ui/setup/env_config/handlers/auto_check_handler.py
Deskripsi: Handler untuk auto check environment
"""

from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

from smartcash.common.utils import is_colab
from smartcash.common.constants.paths import COLAB_PATH
from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger import get_current_ui_logger, create_ui_logger
from smartcash.ui.utils.ui_logger_namespace import ENV_CONFIG_LOGGER_NAMESPACE
from smartcash.ui.setup.env_config.handlers.environment_handler import EnvironmentHandler

class AutoCheckHandler:
    """
    Handler untuk auto check environment
    """
    
    def __init__(self, ui_callback: Optional[Dict[str, Callable]] = None):
        """
        Inisialisasi handler
        
        Args:
            ui_callback: Dictionary callback untuk update UI
        """
        # Coba mendapatkan UI logger yang ada
        self.logger = get_current_ui_logger()
        
        # Jika tidak ada UI logger, gunakan logger dengan namespace env_config
        if not self.logger:
            self.logger = get_logger(ENV_CONFIG_LOGGER_NAMESPACE)
            
        # Set callback functions
        self.ui_callback = ui_callback or {}
        
        # Initialize environment handler
        self.env_handler = EnvironmentHandler(ui_callback)
    
    def _log_message(self, message: str, level: str = "info", icon: str = None):
        """Log message to UI if callback exists"""
        # Log ke logger object
        if level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "success":
            self.logger.info(f"✅ {message}")
        else:
            self.logger.info(message)
            
        # Gunakan callback jika ada
        if 'log_message' in self.ui_callback:
            self.ui_callback['log_message'](message, level, icon)
    
    def _update_status(self, message: str, status_type: str = "info"):
        """Update status in UI if callback exists"""
        if 'update_status' in self.ui_callback:
            self.ui_callback['update_status'](message, status_type)
    
    def _update_progress(self, value: float, message: str = ""):
        """Update progress in UI if callback exists"""
        if 'update_progress' in self.ui_callback:
            self.ui_callback['update_progress'](value, message)
    
    def check_environment(self) -> Dict[str, Any]:
        """
        Check environment and return status
        
        Returns:
            Dict with environment status information
        """
        try:
            # Update progress
            self._update_progress(0.2, "Memeriksa environment...")
            
            # Get environment info
            env_info = {
                'is_colab': is_colab(),
                'base_dir': self.env_handler.env_manager.base_dir,
                'drive_mounted': self.env_handler.env_manager.is_drive_mounted,
                'drive_path': self.env_handler.env_manager.drive_path,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            
            # Log environment info
            self._log_message(f"Memeriksa environment...", "info", "🔍")
            self._log_message(f"Environment: {'Colab' if env_info['is_colab'] else 'Local'}", "info")
            self._log_message(f"Base Directory: {env_info['base_dir']}", "info")
            
            # Check required directories if in Colab
            if env_info['is_colab']:
                required_dirs = self.env_handler.required_dirs
                
                existing_dirs = []
                missing_dirs = []
                
                for dir_name in required_dirs:
                    if Path(f"{COLAB_PATH}/{dir_name}").exists():
                        existing_dirs.append(dir_name)
                    else:
                        missing_dirs.append(dir_name)
                
                env_info['existing_dirs'] = existing_dirs
                env_info['missing_dirs'] = missing_dirs
                
                # Log directory status
                if existing_dirs:
                    self._log_message(f"Direktori yang sudah ada: {', '.join(existing_dirs)}", "success", "✅")
                if missing_dirs:
                    self._log_message(f"Direktori yang belum ada: {', '.join(missing_dirs)}", "warning", "⚠️")
            
            # Update progress
            self._update_progress(1.0, "Pemeriksaan selesai")
            
            # Update status based on check results
            if env_info['is_colab']:
                if not env_info.get('missing_dirs', []):
                    self._update_status("Environment sudah terkonfigurasi", "success")
                    self._log_message("Environment sudah terkonfigurasi", "success", "✅")
                else:
                    self._update_status("Environment perlu dikonfigurasi", "info")
                    self._log_message("Environment perlu dikonfigurasi", "info", "ℹ️")
            else:
                self._update_status("Berjalan di environment lokal", "info")
                self._log_message("Berjalan di environment lokal", "info", "💻")
            
            self._log_message(f"Pemeriksaan environment selesai", "success", "✅")
            
            return env_info
            
        except Exception as e:
            self.logger.error(f"Error saat pemeriksaan environment: {str(e)}")
                
            self._log_message(f"Error saat pemeriksaan environment: {str(e)}", "error", "❌")
            self._update_status(f"Error saat pemeriksaan environment: {str(e)}", "error")
            return {'error': str(e)} 