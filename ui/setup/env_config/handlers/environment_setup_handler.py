"""
File: smartcash/ui/setup/env_config/handlers/environment_setup_handler.py
Deskripsi: Handler untuk setup environment di berbagai platform
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Callable

from smartcash.common.config import get_config_manager
from smartcash.common.utils import is_colab
from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger_namespace import ENV_CONFIG_LOGGER_NAMESPACE
from smartcash.ui.setup.env_config.handlers.local_setup_handler import LocalSetupHandler
from smartcash.ui.setup.env_config.handlers.colab_setup_handler import ColabSetupHandler

class EnvironmentSetupHandler:
    """
    Handler untuk setup environment di berbagai platform
    """
    
    def __init__(self, ui_callback: Optional[Dict[str, Callable]] = None):
        """
        Inisialisasi environment setup handler
        
        Args:
            ui_callback: Dictionary callback untuk update UI
        """
        self.ui_callback = ui_callback or {}
        self.logger = get_logger(ENV_CONFIG_LOGGER_NAMESPACE)
        
        # Inisialisasi handlers berdasarkan platform
        self.local_handler = LocalSetupHandler(ui_callback)
        self.colab_handler = ColabSetupHandler(ui_callback)
    
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
    
    def setup_environment(self) -> Tuple[SimpleConfigManager, Path, Path]:
        """
        Setup environment berdasarkan platform (Colab atau local)
        
        Returns:
            Tuple of (config_manager, base_dir, config_dir)
        """
        # Tentukan direktori berdasarkan jenis platform
        if is_colab():
            self._log_message("Mendeteksi environment Google Colab", "info", "🔍")
            base_dir, config_dir = self.colab_handler.setup_colab_environment()
        else:
            self._log_message("Mendeteksi environment lokal", "info", "🔍")
            base_dir, config_dir = self.local_handler.setup_local_environment()
        
        # Verifikasi bahwa direktori konfigurasi ada dan dapat diakses
        if not config_dir.exists():
            # Jika masih tidak ada, buat direktori kosong sebagai fallback
            config_dir.mkdir(parents=True, exist_ok=True)
            self._log_message("Gagal membuat symlink atau direktori konfigurasi. Menggunakan direktori kosong sebagai fallback.", "warning", "⚠️")
        
        # Initialize config manager
        try:
            config_manager = get_config_manager()
            self._log_message("Config manager berhasil diinisialisasi", "success", "✅")
            self._update_status("Environment berhasil dikonfigurasi", "success")
        except Exception as e:
            self._log_message(f"Error saat inisialisasi config manager: {str(e)}", "error", "❌")
            self._update_status(f"Error saat inisialisasi config manager: {str(e)}", "error")
            raise
        
        return config_manager, base_dir, config_dir

# Fungsi untuk kompatibilitas mundur
def setup_environment(ui_components: Dict[str, Any]):
    """
    Setup environment (fungsi kompatibilitas mundur)
    
    Args:
        ui_components: Dictionary komponen UI, termasuk logger
        
    Returns:
        Tuple of (config_manager, base_dir, config_dir)
    """
    # Setup namespace untuk environment config
    if 'logger_namespace' not in ui_components:
        ui_components['logger_namespace'] = ENV_CONFIG_LOGGER_NAMESPACE
        
    if 'env_config_initialized' not in ui_components:
        ui_components['env_config_initialized'] = True
        
    # Extract logger
    logger = ui_components.get('logger')
    
    # Jika logger tidak tersedia, gunakan logger dengan namespace
    if not logger:
        logger = get_logger(ENV_CONFIG_LOGGER_NAMESPACE)
    
    # Setup UI callbacks
    ui_callbacks = {
        'log_message': lambda msg, level="info", icon=None: logger.info(msg) if logger else None,
        'update_status': lambda msg, status_type: None
    }
    
    # Gunakan log_message callback dari ui_components jika tersedia
    if 'log_message' in ui_components:
        ui_callbacks['log_message'] = ui_components['log_message']
    
    # Gunakan update_status callback dari ui_components jika tersedia
    if 'update_status' in ui_components:
        ui_callbacks['update_status'] = ui_components['update_status']
    
    # Gunakan handler baru dengan callbacks
    env_handler = EnvironmentSetupHandler(ui_callbacks)
    return env_handler.setup_environment() 