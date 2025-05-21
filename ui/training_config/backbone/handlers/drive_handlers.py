"""
File: smartcash/ui/training_config/backbone/handlers/drive_handlers.py
Deskripsi: Handler untuk sinkronisasi konfigurasi backbone dengan Google Drive
"""

from typing import Dict, Any, Optional, Tuple
import os
import ipywidgets as widgets
from pathlib import Path

from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger, LogLevel
from smartcash.common.environment import get_environment_manager
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.training_config.backbone.handlers.config_handlers import update_ui_from_config
from smartcash.ui.training_config.backbone.handlers.sync_logger import update_sync_status_only

# Setup logger dengan level CRITICAL untuk mengurangi log
logger = get_logger(__name__)
logger.set_level(LogLevel.CRITICAL)

def get_default_base_dir():
    """Mendapatkan direktori default berdasarkan environment"""
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def is_colab_environment() -> bool:
    """
    Periksa apakah kode berjalan di lingkungan Google Colab.
    
    Returns:
        Boolean yang menunjukkan apakah kode berjalan di Colab
    """
    return "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ

def is_drive_mounted() -> bool:
    """
    Periksa apakah Google Drive diaktifkan/di-mount.
    
    Returns:
        Boolean yang menunjukkan apakah Google Drive di-mount
    """
    try:
        env_manager = get_environment_manager(base_dir=get_default_base_dir())
        return env_manager.is_drive_mounted
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat memeriksa status Drive: {str(e)}")
        return False

def sync_to_drive(config: Dict[str, Any], ui_components: Dict[str, Any] = None) -> Tuple[bool, str]:
    """
    Sinkronisasi konfigurasi backbone ke Google Drive.
    
    Args:
        config: Konfigurasi yang akan disinkronkan
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Tuple berisi status sukses dan pesan
    """
    try:
        # Update status
        if ui_components:
            update_sync_status_only(ui_components, "Menyinkronkan konfigurasi ke Google Drive...", 'info')
        
        # Jika bukan di Colab, tidak perlu sinkronisasi
        if not is_colab_environment():
            if ui_components:
                update_sync_status_only(ui_components, "Tidak perlu sinkronisasi (bukan di Google Colab)", 'info')
            return True, "Tidak perlu sinkronisasi (bukan di Google Colab)"
        
        # Periksa apakah Google Drive diaktifkan
        if not is_drive_mounted():
            if ui_components:
                update_sync_status_only(ui_components, "Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi.", 'error')
            return False, "Google Drive tidak diaktifkan"
        
        # Dapatkan ConfigManager
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        
        # Sinkronisasi ke Google Drive
        success, message = config_manager.sync_to_drive('model')
        
        if not success:
            logger.error(f"Gagal sinkronisasi backbone ke drive: {message}")
            if ui_components:
                update_sync_status_only(ui_components, f"Gagal menyinkronkan konfigurasi backbone ke Google Drive: {message}", 'error')
            return False, message
        
        if ui_components:
            update_sync_status_only(ui_components, "Konfigurasi backbone berhasil disinkronkan ke Google Drive", 'success')
        logger.info("Konfigurasi backbone berhasil disinkronkan ke Google Drive")
        return True, "Konfigurasi backbone berhasil disinkronkan ke Google Drive"
    except Exception as e:
        error_message = f"Error saat menyinkronkan konfigurasi backbone ke Google Drive: {str(e)}"
        if ui_components:
            update_sync_status_only(ui_components, error_message, 'error')
        logger.error(error_message)
        return False, error_message

def sync_from_drive(ui_components: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Sinkronisasi konfigurasi backbone dari Google Drive.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Tuple berisi status sukses, pesan, dan konfigurasi yang disinkronkan
    """
    try:
        # Update status
        update_sync_status_only(ui_components, "Menyinkronkan konfigurasi dari Google Drive...", 'info')
        
        # Jika bukan di Colab, tidak perlu sinkronisasi
        if not is_colab_environment():
            update_sync_status_only(ui_components, "Tidak perlu sinkronisasi (bukan di Google Colab)", 'info')
            return True, "Tidak perlu sinkronisasi (bukan di Google Colab)", None
        
        # Periksa apakah Google Drive diaktifkan
        if not is_drive_mounted():
            update_sync_status_only(ui_components, "Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi.", 'error')
            return False, "Google Drive tidak diaktifkan", None
        
        # Dapatkan ConfigManager singleton
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        
        # Gunakan sync_with_drive dari ConfigManager
        success, message, drive_config = config_manager.sync_with_drive('model_config.yaml', sync_strategy='drive_priority')
        
        # Log pesan dari sync_with_drive
        if not success:
            logger.error(message)
            update_sync_status_only(ui_components, message, 'error')
            return False, message, None
        
        if drive_config:
            # Simpan konfigurasi ke lokal
            save_success = config_manager.save_module_config('model', drive_config)
            
            if save_success:
                # Update UI dari konfigurasi
                update_ui_from_config(ui_components, drive_config)
                
                # Tampilkan pesan sukses
                update_sync_status_only(ui_components, "Konfigurasi backbone berhasil disinkronkan dari Google Drive", 'success')
                logger.info("Konfigurasi backbone berhasil disinkronkan dari Google Drive")
                return True, "Konfigurasi backbone berhasil disinkronkan dari Google Drive", drive_config
            else:
                # Tampilkan pesan error
                update_sync_status_only(ui_components, "Gagal menyimpan konfigurasi backbone dari Google Drive", 'error')
                logger.error("Gagal menyimpan konfigurasi backbone dari Google Drive")
                return False, "Gagal menyimpan konfigurasi backbone dari Google Drive", None
        else:
            # Tampilkan pesan error
            update_sync_status_only(ui_components, "Gagal memuat konfigurasi backbone dari Google Drive", 'error')
            logger.error("Gagal memuat konfigurasi backbone dari Google Drive")
            return False, "Gagal memuat konfigurasi backbone dari Google Drive", None
    except Exception as e:
        error_message = f"Error saat menyinkronkan konfigurasi backbone dari Google Drive: {str(e)}"
        update_sync_status_only(ui_components, error_message, 'error')
        logger.error(error_message)
        return False, error_message, None

def sync_with_drive(config: Dict[str, Any], ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Sinkronisasi konfigurasi dengan Google Drive.
    
    Args:
        config: Konfigurasi yang akan disinkronkan
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Konfigurasi yang telah disinkronkan
    """
    # Jika bukan di lingkungan Colab, tidak perlu sinkronisasi
    if not is_colab_environment():
        if ui_components:
            update_sync_status_only(ui_components, "Tidak perlu sinkronisasi (bukan di Google Colab)", 'info')
        return config
        
    # Jika Google Drive tidak diaktifkan, tidak perlu sinkronisasi
    if not is_drive_mounted():
        if ui_components:
            update_sync_status_only(ui_components, "Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi.", 'warning')
        return config
        
    # Coba sinkronisasi ke Google Drive
    success, message = sync_to_drive(config, ui_components)
    
    # Kembalikan konfigurasi asli jika gagal
    if not success:
        return config
        
    # Kembalikan konfigurasi yang disinkronkan
    config_manager = get_config_manager(base_dir=get_default_base_dir())
    return config_manager.get_config('model', reload=True)