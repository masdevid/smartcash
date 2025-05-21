"""
File: smartcash/ui/training_config/backbone/handlers/save_handlers.py
Deskripsi: Handler untuk menyimpan konfigurasi backbone model
"""

from typing import Dict, Any
import os
from pathlib import Path

from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger, LogLevel
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.training_config.backbone.handlers.drive_handlers import sync_with_drive

# Setup logger dengan level info untuk mengurangi log
logger = get_logger(__name__)
logger.set_level(LogLevel.INFO)

def get_default_base_dir():
    """Mendapatkan direktori default berdasarkan environment"""
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def save_backbone_config(config: Dict[str, Any], ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Simpan konfigurasi backbone dan sinkronisasi dengan Google Drive jika diperlukan.
    
    Args:
        config: Dictionary konfigurasi yang akan disimpan
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Konfigurasi yang telah disimpan
    """
    try:
        # Update status panel jika ui_components tersedia
        if ui_components:
            from smartcash.ui.training_config.backbone.handlers.sync_logger import update_sync_status_only
            update_sync_status_only(ui_components, "Menyimpan konfigurasi backbone...", 'info')
        
        # Simpan konfigurasi
        saved_config = _save_local_config(config)
        
        # Sinkronisasi dengan Google Drive jika diperlukan
        synced_config = sync_with_drive(saved_config, ui_components)
        
        return synced_config
    except Exception as e:
        error_message = f"Error saat menyimpan konfigurasi backbone: {str(e)}"
        logger.error(f"{ICONS.get('error', '❌')} {error_message}")
        
        # Update status panel jika ui_components tersedia
        if ui_components:
            from smartcash.ui.training_config.backbone.handlers.sync_logger import update_sync_status_only
            update_sync_status_only(ui_components, error_message, 'error')
        
        return config

def _save_local_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simpan konfigurasi backbone ke storage lokal.
    
    Args:
        config: Dictionary konfigurasi yang akan disimpan
        
    Returns:
        Konfigurasi yang telah disimpan
    """
    try:
        # Pastikan konfigurasi memiliki struktur yang benar
        original_config = config.copy() if config else {}
        
        if 'model' not in original_config:
            original_config = {'model': original_config}
        
        # Simpan konfigurasi
        base_dir = get_default_base_dir()
        config_manager = get_config_manager(base_dir=base_dir)
        save_success = config_manager.save_config(original_config, 'model')
        
        if not save_success:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menyimpan konfigurasi backbone")
            return original_config
        
        logger.info(f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil disimpan")
        
        # Kembalikan konfigurasi yang telah disimpan
        return config_manager.get_config('model', reload=True)
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi backbone: {str(e)}")
        return config