"""
File: smartcash/ui/training_config/backbone/handlers/drive_handlers.py
Deskripsi: Handler untuk sinkronisasi konfigurasi backbone dengan Google Drive
"""

from typing import Dict, Any, Optional
import os
import ipywidgets as widgets
from IPython.display import clear_output, display
from pathlib import Path

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert, create_status_indicator
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger, LogLevel
from smartcash.common.environment import get_environment_manager
from smartcash.ui.training_config.backbone.handlers.config_handlers import update_ui_from_config

# Setup logger dengan level CRITICAL untuk mengurangi log
logger = get_logger(__name__)
logger.set_level(LogLevel.CRITICAL)

def get_default_base_dir():
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def sync_to_drive(button: Optional[widgets.Button], ui_components: Dict[str, Any]) -> None:
    """
    Sinkronisasi konfigurasi backbone ke Google Drive.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary berisi komponen UI
    """
    status_panel = ui_components.get('status_panel')
    if not status_panel:
        logger.error("Status panel tidak ditemukan")
        return
    
    with status_panel:
        clear_output(wait=True)
        try:
            # Dapatkan environment manager
            env_manager = get_environment_manager(base_dir=get_default_base_dir())
            
            # Cek apakah drive diaktifkan
            if not env_manager.is_drive_mounted:
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi.",
                        alert_type='error'
                    ))
                return
            
            # Dapatkan ConfigManager singleton
            config_manager = get_config_manager(base_dir=get_default_base_dir())
            
            # Gunakan sync_to_drive dari ConfigManager yang baru
            success, message = config_manager.sync_to_drive('model')
            
            # Log pesan dari sync_to_drive
            if not success:
                logger.error(message)
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} {message}",
                        alert_type='error'
                    ))
                return
            
            # Tampilkan pesan sukses
            with status_panel:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil disinkronkan ke Google Drive",
                    alert_type='success'
                ))
            
            logger.info("Konfigurasi backbone berhasil disinkronkan ke Google Drive")
        except Exception as e:
            # Tampilkan pesan error
            with status_panel:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS.get('error', '❌')} Error saat menyinkronkan konfigurasi backbone ke Google Drive: {str(e)}",
                    alert_type='error'
                ))
            
            logger.error(f"Error saat menyinkronkan konfigurasi backbone ke Google Drive: {str(e)}")

def sync_from_drive(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Sinkronisasi konfigurasi backbone dari Google Drive.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary berisi komponen UI
    """
    status_panel = ui_components.get('status_panel')
    if not status_panel:
        logger.error("Status panel tidak ditemukan")
        return
    
    with status_panel:
        clear_output(wait=True)
        try:
            # Dapatkan environment manager
            env_manager = get_environment_manager(base_dir=get_default_base_dir())
            
            # Cek apakah drive diaktifkan
            if not env_manager.is_drive_mounted:
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi.",
                        alert_type='error'
                    ))
                return
            
            # Dapatkan ConfigManager singleton
            config_manager = get_config_manager(base_dir=get_default_base_dir())
            
            # Gunakan sync_with_drive dari ConfigManager yang baru
            success, message, drive_config = config_manager.sync_with_drive('model_config.yaml', sync_strategy='drive_priority')
            
            # Log pesan dari sync_with_drive
            if not success:
                logger.error(message)
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} {message}",
                        alert_type='error'
                    ))
                return
            
            if drive_config:
                # Simpan konfigurasi ke lokal
                success = config_manager.save_module_config('model', drive_config)
                
                if success:
                    # Update UI dari konfigurasi
                    update_ui_from_config(ui_components, drive_config)
                    
                    # Tampilkan pesan sukses
                    with status_panel:
                        clear_output(wait=True)
                        display(create_info_alert(
                            f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil disinkronkan dari Google Drive",
                            alert_type='success'
                        ))
                    
                    logger.info("Konfigurasi backbone berhasil disinkronkan dari Google Drive")
                else:
                    # Tampilkan pesan error
                    with status_panel:
                        clear_output(wait=True)
                        display(create_info_alert(
                            f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi backbone dari Google Drive",
                            alert_type='error'
                        ))
                    
                    logger.error("Gagal menyimpan konfigurasi backbone dari Google Drive")
            else:
                # Tampilkan pesan error
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Gagal memuat konfigurasi backbone dari Google Drive",
                        alert_type='error'
                    ))
                
                logger.error("Gagal memuat konfigurasi backbone dari Google Drive")
        except Exception as e:
            # Tampilkan pesan error
            with status_panel:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS.get('error', '❌')} Error saat menyinkronkan konfigurasi backbone dari Google Drive: {str(e)}",
                    alert_type='error'
                ))
            
            logger.error(f"Error saat menyinkronkan konfigurasi backbone dari Google Drive: {str(e)}")
