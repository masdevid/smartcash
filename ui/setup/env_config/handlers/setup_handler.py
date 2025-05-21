"""
File: smartcash/ui/setup/env_config/handlers/setup_handler.py
Deskripsi: Handler untuk setup environment
"""

from typing import Dict, Any, Tuple, Optional, Callable
from pathlib import Path
import logging

from smartcash.common.config.manager import SimpleConfigManager, get_config_manager
from smartcash.ui.setup.env_config.handlers.environment_setup_handler import EnvironmentSetupHandler
from smartcash.ui.setup.env_config.handlers.config_info_handler import display_config_info

class SetupHandler:
    """
    Handler utama untuk setup environment
    """
    
    def __init__(self, ui_callback: Optional[Dict[str, Callable]] = None):
        """
        Inisialisasi setup handler
        
        Args:
            ui_callback: Dictionary callback untuk update UI
        """
        self.ui_callback = ui_callback or {}
        self.logger = logging.getLogger(__name__)
        self.env_setup_handler = EnvironmentSetupHandler(ui_callback)
    
    def _log_message(self, message: str):
        """Log message to UI if callback exists"""
        self.logger.info(message)
        if 'log_message' in self.ui_callback:
            self.ui_callback['log_message'](message)
    
    def _update_status(self, message: str, status_type: str = "info"):
        """Update status in UI if callback exists"""
        if 'update_status' in self.ui_callback:
            self.ui_callback['update_status'](message, status_type)
    
    def perform_setup(self) -> Tuple[SimpleConfigManager, Path, Path]:
        """
        Melakukan setup environment
        
        Returns:
            Tuple berisi (config_manager, base_dir, config_dir)
        """
        try:
            # Setup environment dan dapatkan informasi direktori
            self._log_message("🚀 Memulai setup environment")
            self._update_status("Setup environment sedang berjalan...", "info")
            
            config_manager, base_dir, config_dir = self.env_setup_handler.setup_environment()
            
            self._log_message(f"✅ Setup environment berhasil. Base dir: {base_dir}, Config dir: {config_dir}")
            self._update_status("Setup environment berhasil", "success")
            
            return config_manager, base_dir, config_dir
            
        except Exception as e:
            self._log_message(f"❌ Error saat setup environment: {str(e)}")
            self._update_status(f"Error saat setup environment: {str(e)}", "error")
            raise
    
    def handle_error(self, error: Exception) -> SimpleConfigManager:
        """
        Menangani error saat setup
        
        Args:
            error: Exception yang terjadi
            
        Returns:
            Config manager (fallback)
        """
        self._log_message(f"❌ Error saat inisialisasi environment: {str(error)}")
        self._update_status(f"Error saat inisialisasi environment: {str(error)}", "error")
        
        # Coba dapatkan config manager sebagai fallback
        try:
            config_manager = get_config_manager()
            self._log_message("✅ Berhasil mendapatkan config manager fallback")
            return config_manager
        except Exception as e:
            self._log_message(f"❌ Error mendapatkan config manager fallback: {str(e)}")
            return None

# Fungsi untuk kompatibilitas mundur
def perform_setup(ui_components: Dict[str, Any]) -> Tuple[SimpleConfigManager, Path, Path]:
    """
    Melakukan setup environment (fungsi kompatibilitas mundur)
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Tuple berisi (config_manager, base_dir, config_dir)
    """
    # Extract callbacks
    ui_callbacks = {}
    
    # Gunakan log_message callback dari ui_components jika tersedia
    if 'log_message' in ui_components:
        ui_callbacks['log_message'] = ui_components['log_message']
    
    # Gunakan update_status callback dari ui_components jika tersedia
    if 'update_status' in ui_components:
        ui_callbacks['update_status'] = ui_components['update_status']
    
    # Setup environment dan dapatkan informasi direktori
    handler = SetupHandler(ui_callbacks)
    config_manager, base_dir, config_dir = handler.perform_setup()
    
    # Tampilkan informasi konfigurasi
    display_config_info(ui_components, config_manager, base_dir, config_dir)
    
    return config_manager, base_dir, config_dir

def handle_setup_error(ui_components: Dict[str, Any], error: Exception) -> SimpleConfigManager:
    """
    Menangani error saat setup (fungsi kompatibilitas mundur)
    
    Args:
        ui_components: Dictionary komponen UI
        error: Exception yang terjadi
        
    Returns:
        Config manager (fallback)
    """
    # Extract callbacks
    ui_callbacks = {}
    
    # Gunakan log_message callback dari ui_components jika tersedia
    if 'log_message' in ui_components:
        ui_callbacks['log_message'] = ui_components['log_message']
    
    # Gunakan update_status callback dari ui_components jika tersedia
    if 'update_status' in ui_components:
        ui_callbacks['update_status'] = ui_components['update_status']
    
    # Tangani error
    handler = SetupHandler(ui_callbacks)
    return handler.handle_error(error) 