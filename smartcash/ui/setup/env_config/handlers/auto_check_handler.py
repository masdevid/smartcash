"""
File: smartcash/ui/setup/env_config/handlers/auto_check_handler.py
Deskripsi: Handler untuk auto check environment - disederhanakan dan mengurangi duplikasi
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from smartcash.ui.setup.env_config.handlers.base_handler import BaseHandler, EnvConfigHandlerMixin

class AutoCheckHandler(BaseHandler, EnvConfigHandlerMixin):
    """
    Handler untuk auto check environment - fokus pada checking dan validation
    """
    
    def __init__(self, ui_components: Dict[str, Any] = None, ui_callback: Optional[Dict[str, Callable]] = None):
        """
        Inisialisasi handler dengan UI components atau callbacks
        
        Args:
            ui_components: Dictionary komponen UI (preferred)
            ui_callback: Dictionary callback untuk update UI (legacy support)
        """
        super().__init__(ui_components, ui_callback)
    
    def check_environment(self) -> Dict[str, Any]:
        """
        Check environment and return status dengan progress tracking
        """
        try:
            self._update_progress(0.1, "Memeriksa environment...")
            
            # Validate environment setup
            env_status = self.validate_environment_setup()
            
            # Add timestamp
            env_status['timestamp'] = datetime.now().strftime('%H:%M:%S')
            
            # Log environment info
            self._log_environment_info(env_status)
            
            # Update progress and status
            self._update_progress(1.0, "Pemeriksaan selesai")
            self._update_final_status(env_status)
            
            return env_status
            
        except Exception as e:
            error_msg = f"Error saat pemeriksaan environment: {str(e)}"
            self._log_message(error_msg, "error", "❌")
            self._update_status(error_msg, "error")
            return {'error': str(e)}
    
    def _log_environment_info(self, env_status: Dict[str, Any]):
        """Log informasi environment berdasarkan status"""
        # Basic environment info
        env_type = 'Colab' if env_status.get('is_colab', False) else 'Local'
        self._log_message(f"Environment: {env_type} 💻", "info", "🔍")
        
        if 'base_dir' in env_status:
            self._log_message(f"Base Directory: {env_status['base_dir']} 📁", "info")
        
        # Drive connection status
        drive_connected = env_status.get('drive_connected', False)
        if env_status.get('is_colab'):
            drive_icon = "✅" if drive_connected else "❌"
            drive_status = "Terhubung" if drive_connected else "Tidak terhubung"
            self._log_message(f"Google Drive: {drive_status} {drive_icon}", 
                            "success" if drive_connected else "warning")
        
        # Directory status
        directories_exist = env_status.get('directories_exist', {})
        if directories_exist:
            existing_dirs = [k for k, v in directories_exist.items() if v]
            missing_dirs = [k for k, v in directories_exist.items() if not v]
            
            if existing_dirs:
                self._log_message(f"Direktori tersedia: {', '.join(existing_dirs)} ✅", "success")
            if missing_dirs:
                self._log_message(f"Direktori belum ada: {', '.join(missing_dirs)} ⚠️", "warning")
    
    def _update_final_status(self, env_status: Dict[str, Any]):
        """Update status akhir berdasarkan hasil pemeriksaan"""
        if env_status.get('error'):
            return
            
        if env_status.get('ready', False):
            self._update_status("Environment sudah terkonfigurasi", "success")
            self._log_message("Environment sudah terkonfigurasi 🎉", "success", "✅")
        elif env_status.get('is_colab', False):
            self._update_status("Environment perlu dikonfigurasi", "info")
            self._log_message("Environment perlu dikonfigurasi 🔧", "info", "ℹ️")
        else:
            self._update_status("Berjalan di environment lokal", "info")
            self._log_message("Berjalan di environment lokal 🏠", "info", "💻")