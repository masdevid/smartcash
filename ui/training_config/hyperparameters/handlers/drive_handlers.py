"""
File: smartcash/ui/training_config/hyperparameters/handlers/drive_handlers.py
Deskripsi: Handler untuk sinkronisasi konfigurasi hyperparameters dengan Google Drive
"""

from typing import Dict, Any, Optional, Tuple
import os
import ipywidgets as widgets
from pathlib import Path

from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger, LogLevel
from smartcash.common.environment import get_environment_manager
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.training_config.hyperparameters.handlers.config_writer import update_ui_from_config
from smartcash.ui.training_config.hyperparameters.handlers.config_manager import get_default_base_dir
from smartcash.ui.training_config.hyperparameters.handlers.sync_logger import (
    update_sync_status_only,
    log_sync_success,
    log_sync_error,
    log_sync_warning
)

# Setup logger dengan level INFO
logger = get_logger(__name__)
logger.set_level(LogLevel.INFO)

def is_colab_environment() -> bool:
    """
    Periksa apakah kode berjalan di lingkungan Google Colab.
    
    Returns:
        Boolean yang menunjukkan apakah kode berjalan di Colab
    """
    return "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ

def sync_to_drive(button: Optional[widgets.Button], ui_components: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Sinkronisasi konfigurasi hyperparameters ke Google Drive.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        Tuple berisi status sukses dan pesan
    """
    try:
        # Update status
        update_sync_status_only(ui_components, "Menyinkronkan konfigurasi ke Google Drive...", 'info')
        
        # Periksa apakah di lingkungan Colab
        if not is_colab_environment():
            update_sync_status_only(ui_components, "Tidak perlu sinkronisasi (bukan di Google Colab)", 'info')
            return True, "Tidak perlu sinkronisasi (bukan di Google Colab)"
        
        # Periksa apakah Google Drive diaktifkan
        env_manager = get_environment_manager(base_dir=get_default_base_dir())
        if not env_manager.is_drive_mounted:
            update_sync_status_only(ui_components, "Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi.", 'error')
            return False, "Google Drive tidak diaktifkan"
        
        # Dapatkan ConfigManager
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        
        # Sinkronisasi ke Google Drive
        success, message = config_manager.sync_to_drive('hyperparameters')
        
        if not success:
            log_sync_error(ui_components, f"Gagal sinkronisasi hyperparameters ke drive: {message}")
            return False, message
        
        log_sync_success(ui_components, "Konfigurasi hyperparameters berhasil disinkronkan ke Google Drive")
        return True, "Konfigurasi hyperparameters berhasil disinkronkan ke Google Drive"
    except Exception as e:
        error_message = f"Error saat menyinkronkan konfigurasi hyperparameters ke Google Drive: {str(e)}"
        log_sync_error(ui_components, error_message)
        return False, error_message

def sync_from_drive(button: Optional[widgets.Button], ui_components: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Sinkronisasi konfigurasi hyperparameters dari Google Drive.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        Tuple berisi status sukses, pesan, dan konfigurasi yang disinkronkan
    """
    try:
        # Update status
        update_sync_status_only(ui_components, "Menyinkronkan konfigurasi dari Google Drive...", 'info')
        
        # Periksa apakah di lingkungan Colab
        if not is_colab_environment():
            update_sync_status_only(ui_components, "Tidak perlu sinkronisasi (bukan di Google Colab)", 'info')
            return True, "Tidak perlu sinkronisasi (bukan di Google Colab)", None
        
        # Periksa apakah Google Drive diaktifkan
        env_manager = get_environment_manager(base_dir=get_default_base_dir())
        if not env_manager.is_drive_mounted:
            update_sync_status_only(ui_components, "Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi.", 'error')
            return False, "Google Drive tidak diaktifkan", None
        
        # Dapatkan ConfigManager singleton
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        
        # Gunakan sync_with_drive dari ConfigManager
        success, message, drive_config = config_manager.sync_with_drive('hyperparameters_config.yaml', sync_strategy='drive_priority')
        
        # Log pesan dari sync_with_drive
        if not success:
            log_sync_error(ui_components, message)
            return False, message, None
        
        if drive_config:
            # Simpan konfigurasi ke lokal
            save_success = config_manager.save_module_config('hyperparameters', drive_config)
            
            if save_success:
                # Update UI dari konfigurasi
                update_ui_from_config(ui_components, drive_config)
                
                # Tampilkan pesan sukses
                log_sync_success(ui_components, "Konfigurasi hyperparameters berhasil disinkronkan dari Google Drive")
                return True, "Konfigurasi hyperparameters berhasil disinkronkan dari Google Drive", drive_config
            else:
                # Tampilkan pesan error
                log_sync_error(ui_components, "Gagal menyimpan konfigurasi hyperparameters dari Google Drive")
                return False, "Gagal menyimpan konfigurasi hyperparameters dari Google Drive", None
        else:
            # Tampilkan pesan error
            log_sync_error(ui_components, "Gagal memuat konfigurasi hyperparameters dari Google Drive")
            return False, "Gagal memuat konfigurasi hyperparameters dari Google Drive", None
    except Exception as e:
        error_message = f"Error saat menyinkronkan konfigurasi hyperparameters dari Google Drive: {str(e)}"
        log_sync_error(ui_components, error_message)
        return False, error_message, None