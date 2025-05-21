"""
File: smartcash/ui/setup/env_config/handlers/environment_setup_handler.py
Deskripsi: Handler untuk setup environment di berbagai platform
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Callable
import logging

from smartcash.common.config.manager import SimpleConfigManager, get_config_manager
from smartcash.common.utils import is_colab
from smartcash.ui.setup.env_config.utils.fallback_logger import get_fallback_logger
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
        self.logger = logging.getLogger(__name__)
        
        # Inisialisasi handlers berdasarkan platform
        self.local_handler = LocalSetupHandler(ui_callback)
        self.colab_handler = ColabSetupHandler(ui_callback)
    
    def _log_message(self, message: str):
        """Log message to UI if callback exists"""
        self.logger.info(message)
        if 'log_message' in self.ui_callback:
            self.ui_callback['log_message'](message)
    
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
            self._log_message("🔍 Mendeteksi environment Google Colab")
            base_dir, config_dir = self.colab_handler.setup_colab_environment()
        else:
            self._log_message("🔍 Mendeteksi environment lokal")
            base_dir, config_dir = self.local_handler.setup_local_environment()
        
        # Verifikasi bahwa direktori konfigurasi ada dan dapat diakses
        if not config_dir.exists():
            # Jika masih tidak ada, buat direktori kosong sebagai fallback
            config_dir.mkdir(parents=True, exist_ok=True)
            self._log_message("⚠️ Gagal membuat symlink atau direktori konfigurasi. Menggunakan direktori kosong sebagai fallback.")
        
        # Initialize config manager
        try:
            config_manager = get_config_manager()
            self._log_message("✅ Config manager berhasil diinisialisasi")
            self._update_status("Environment berhasil dikonfigurasi", "success")
        except Exception as e:
            self._log_message(f"❌ Error saat inisialisasi config manager: {str(e)}")
            self._update_status(f"Error saat inisialisasi config manager: {str(e)}", "error")
            raise
        
        return config_manager, base_dir, config_dir

# Fungsi untuk kompatibilitas mundur
def setup_environment(ui_components: Dict[str, Any]) -> Tuple[SimpleConfigManager, Path, Path]:
    """
    Setup environment (fungsi kompatibilitas mundur)
    
    Args:
        ui_components: Dictionary komponen UI, termasuk logger
        
    Returns:
        Tuple of (config_manager, base_dir, config_dir)
    """
    # Extract logger and create ui_callbacks
    logger = ui_components.get('logger', None)
    
    # Jika logger tidak tersedia, gunakan fallback
    if not logger:
        logger = get_fallback_logger("env_config_setup")
    
    # Setup UI callbacks
    ui_callbacks = {
        'log_message': lambda msg: logger.info(msg) if logger else None,
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