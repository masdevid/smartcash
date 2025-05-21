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
from smartcash.ui.utils.ui_logger_namespace import ENV_CONFIG_LOGGER_NAMESPACE
from smartcash.ui.setup.env_config.handlers.base_handler import BaseHandler
from smartcash.ui.setup.env_config.handlers.environment_handler import EnvironmentHandler

class AutoCheckHandler(BaseHandler):
    """
    Handler untuk auto check environment
    """
    
    def __init__(self, ui_callback: Optional[Dict[str, Callable]] = None):
        """
        Inisialisasi handler
        
        Args:
            ui_callback: Dictionary callback untuk update UI
        """
        super().__init__(ui_callback, ENV_CONFIG_LOGGER_NAMESPACE)
        
        # Initialize environment handler
        self.env_handler = EnvironmentHandler(ui_callback)
    
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