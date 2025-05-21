"""
File: smartcash/ui/dataset/split/handlers/reset_handlers.py
Deskripsi: Handler untuk reset konfigurasi di split dataset
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path
from smartcash.common.logger import get_logger
from smartcash.common.constants.log_messages import OPERATION_SUCCESS, OPERATION_FAILED, CONFIG_LOADED
from smartcash.common.config import get_config_manager
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.dataset.split.handlers.config_handlers import (
    load_config, save_config, update_ui_from_config, get_default_split_config, is_colab_environment
)
from smartcash.ui.dataset.split.handlers.status_handlers import update_status_panel

logger = get_logger(__name__)

def get_default_base_dir():
    """Dapatkan direktori base default."""
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def handle_reset_action(ui_components: Dict[str, Any]) -> None:
    """
    Handle aksi reset konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Update status
        update_status_panel(ui_components, "Mereset konfigurasi split dataset...", 'info')
        
        # Dapatkan konfigurasi default
        default_config = get_default_split_config()
        
        # Update UI dari konfigurasi default
        update_ui_from_config(ui_components, default_config)
        
        # Simpan konfigurasi default ke config manager
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        save_success = config_manager.save_module_config('split', default_config)
        
        if not save_success:
            update_status_panel(ui_components, "Gagal menyimpan konfigurasi default", 'error')
            logger.error(f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi default")
            return
        
        # Sinkronisasi dengan Google Drive jika di Colab
        drive_message = ""
        if is_colab_environment():
            try:
                # Cek apakah drive terpasang
                env_manager = get_environment_manager()
                if env_manager.is_drive_mounted:
                    # Sinkronisasi konfigurasi
                    success, message = config_manager.sync_to_drive('split')
                    if success:
                        drive_message = " dan disinkronkan dengan Google Drive"
                    else:
                        logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal sinkronisasi dengan Google Drive: {message}")
                        update_status_panel(ui_components, f"Konfigurasi direset, tetapi gagal sinkronisasi: {message}", 'warning')
                        return
                else:
                    logger.info(f"{ICONS.get('info', 'ℹ️')} Google Drive tidak terpasang, skip sinkronisasi")
            except Exception as e:
                logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat sinkronisasi: {str(e)}")
        
        # Verifikasi konfigurasi tersimpan dengan benar
        saved_config = load_config()
        
        # Periksa apakah sesuai dengan default
        is_consistent = True
        if 'split' in saved_config and 'split' in default_config:
            for key, value in default_config['split'].items():
                if key not in saved_config['split'] or saved_config['split'][key] != value:
                    is_consistent = False
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Konfigurasi reset tidak konsisten pada key '{key}'")
                    break
        
        if is_consistent:
            update_status_panel(ui_components, f"Konfigurasi split dataset berhasil direset ke default{drive_message}", 'success')
            logger.info(OPERATION_SUCCESS.format(operation="Reset konfigurasi"))
        else:
            update_status_panel(ui_components, "Konfigurasi direset tetapi tidak konsisten dengan default", 'warning')
            logger.warning(f"{ICONS.get('warning', '⚠️')} Konfigurasi direset tetapi tidak konsisten dengan default")
        
    except Exception as e:
        # Update status
        error_message = f"Error saat reset konfigurasi: {str(e)}"
        update_status_panel(ui_components, error_message, 'error')
        logger.error(OPERATION_FAILED.format(operation="Reset konfigurasi", reason=str(e)))

def create_reset_handler(ui_components: Dict[str, Any]):
    """
    Buat handler untuk tombol reset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Function handler untuk tombol reset
    """
    def on_reset_clicked(b):
        handle_reset_action(ui_components)
    
    return on_reset_clicked

# Import yang diperlukan untuk sinkronisasi
try:
    from smartcash.common.environment import get_environment_manager
except ImportError:
    # Fallback jika module tidak tersedia
    def get_environment_manager(*args, **kwargs):
        class DummyEnvironmentManager:
            def __init__(self):
                self.is_drive_mounted = False
        return DummyEnvironmentManager()
    logger.warning(f"{ICONS.get('warning', '⚠️')} smartcash.common.environment tidak tersedia, menggunakan dummy")