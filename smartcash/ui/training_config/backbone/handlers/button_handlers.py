"""
File: smartcash/ui/training_config/backbone/handlers/button_handlers.py
Deskripsi: Handler untuk tombol pada UI pemilihan backbone model SmartCash
"""

from typing import Dict, Any, Callable
import ipywidgets as widgets
from IPython.display import clear_output, display
from pathlib import Path
import os

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert, create_status_indicator
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger, LogLevel
from smartcash.common.environment import get_environment_manager
from smartcash.ui.training_config.backbone.handlers.config_handlers import (
    update_config_from_ui,
    update_ui_from_config,
    update_backbone_info
)
from smartcash.ui.training_config.backbone.handlers.drive_handlers import sync_to_drive

# Setup logger dengan level CRITICAL untuk mengurangi log
logger = get_logger(__name__)
logger.set_level(LogLevel.CRITICAL)

def get_default_base_dir():
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def on_save_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol Save pada UI backbone.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary berisi komponen UI
    """
    status_panel = ui_components.get('status_panel')
    if not status_panel:
        logger.error(f"{ICONS.get('error', '❌')} Status panel tidak ditemukan")
        return
    
    with status_panel:
        clear_output(wait=True)
        display(create_status_indicator('info', f"{ICONS.get('info', 'ℹ️')} Menyimpan konfigurasi backbone..."))
    
    try:
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        
        # Update config dari UI
        config_to_save = update_config_from_ui(ui_components)
        
        # Simpan config ke file
        success = config_manager.save_module_config('model', config_to_save)
        
        # Memastikan persistensi UI dengan notifikasi observer
        config_manager.ensure_ui_persistence('model', config_to_save)
        
        # Pastikan UI components teregistrasi untuk persistensi
        try:
            config_manager.register_ui_components('backbone', ui_components)
        except Exception as persist_error:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat memastikan persistensi UI: {persist_error}")
        
        # Tampilkan pesan sukses atau warning
        with status_panel:
            clear_output(wait=True)
            if success:
                display(create_info_alert(
                    f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil disimpan",
                    alert_type='success'
                ))
            else:
                display(create_info_alert(
                    f"{ICONS.get('warning', '⚠️')} Konfigurasi backbone mungkin tidak tersimpan ke file",
                    alert_type='warning'
                ))
        
        # Update info panel
        update_backbone_info(ui_components)
        
        # Sinkronisasi ke Google Drive jika diaktifkan
        try:
            env_manager = get_environment_manager()
            if env_manager.is_drive_mounted:
                # Sinkronisasi ke Google Drive
                logger.info(f"{ICONS.get('info', 'ℹ️')} Menyinkronkan konfigurasi backbone ke Google Drive...")
                sync_to_drive(None, ui_components)
        except Exception as e:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menyinkronkan ke Google Drive: {str(e)}")
        
        logger.critical(f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil disimpan")
    except Exception as e:
        # Tampilkan pesan error
        with status_panel:
            clear_output(wait=True)
            display(create_info_alert(
                f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi: {str(e)}",
                alert_type='error'
            ))
        
        logger.critical(f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi: {str(e)}")

def on_reset_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol Reset pada UI backbone.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary berisi komponen UI
    """
    status_panel = ui_components.get('status_panel')
    if not status_panel:
        logger.error(f"{ICONS.get('error', '❌')} Status panel tidak ditemukan")
        return
    
    with status_panel:
        clear_output(wait=True)
        display(create_status_indicator('info', f"{ICONS.get('info', 'ℹ️')} Mereset konfigurasi backbone..."))
    
    try:
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        
        # Dapatkan default config
        default_config = config_manager.get_module_config('model', {})
        
        # Update UI dari default config
        update_ui_from_config(ui_components, default_config)
        
        # Simpan default config ke file
        success = config_manager.save_module_config('model', default_config)
        
        # Memastikan persistensi UI dengan notifikasi observer
        config_manager.ensure_ui_persistence('model', default_config)
        
        # Pastikan UI components teregistrasi untuk persistensi
        try:
            config_manager.register_ui_components('backbone', ui_components)
        except Exception as persist_error:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat memastikan persistensi UI: {persist_error}")
        
        # Tampilkan pesan sukses atau warning
        with status_panel:
            clear_output(wait=True)
            if success:
                display(create_info_alert(
                    f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil direset ke default",
                    alert_type='success'
                ))
            else:
                display(create_info_alert(
                    f"{ICONS.get('warning', '⚠️')} Konfigurasi backbone direset di UI tetapi mungkin tidak tersimpan ke file",
                    alert_type='warning'
                ))
        
        # Update info panel
        update_backbone_info(ui_components)
        
        # Sinkronisasi ke Google Drive jika diaktifkan
        try:
            env_manager = get_environment_manager()
            if env_manager.is_drive_mounted:
                # Sinkronisasi ke Google Drive
                logger.info(f"{ICONS.get('info', 'ℹ️')} Menyinkronkan konfigurasi backbone ke Google Drive...")
                sync_to_drive(None, ui_components)
        except Exception as e:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menyinkronkan ke Google Drive: {str(e)}")
        
        logger.critical(f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil direset ke default")
    except Exception as e:
        # Tampilkan pesan error
        with status_panel:
            clear_output(wait=True)
            display(create_info_alert(
                f"{ICONS.get('error', '❌')} Gagal mereset konfigurasi: {str(e)}",
                alert_type='error'
            ))
        
        logger.critical(f"{ICONS.get('error', '❌')} Gagal mereset konfigurasi: {str(e)}")
