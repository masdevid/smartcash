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
from smartcash.common.environment import get_environment_manager, get_default_base_dir
from smartcash.ui.training_config.backbone.handlers.config_handlers import (
    update_config_from_ui,
    update_ui_from_config,
    update_backbone_info,
    get_default_backbone_config
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
    Handler untuk tombol save.
    
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
            # Dapatkan ConfigManager singleton
            config_manager = get_config_manager(base_dir=get_default_base_dir())
            
            # Update konfigurasi dari UI
            config = update_config_from_ui(ui_components)
            
            # Simpan konfigurasi
            success = config_manager.save_module_config('model', config)
            
            if not success:
                # Tampilkan pesan error
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi backbone",
                        alert_type='error'
                    ))
                return
            
            # Update info panel
            update_backbone_info(ui_components)
            
            # Sinkronisasi ke Google Drive jika diaktifkan
            try:
                env_manager = get_environment_manager()
                if env_manager.is_drive_mounted:
                    # Sinkronisasi ke Google Drive
                    logger.info("Menyinkronkan konfigurasi backbone ke Google Drive...")
                    sync_to_drive(None, ui_components)
            except Exception as e:
                logger.warning(f"Gagal menyinkronkan ke Google Drive: {str(e)}")
            
            logger.info("Konfigurasi backbone berhasil disimpan")
        except Exception as e:
            # Tampilkan pesan error
            with status_panel:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi: {str(e)}",
                    alert_type='error'
                ))
            
            logger.error(f"Gagal menyimpan konfigurasi: {str(e)}")

def on_reset_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol reset.
    
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
            # Dapatkan ConfigManager singleton
            config_manager = get_config_manager(base_dir=get_default_base_dir())
            
            # Reset konfigurasi ke default
            config = get_default_backbone_config()
            
            # Simpan konfigurasi default
            success = config_manager.save_module_config('backbone', config)
            
            if not success:
                # Tampilkan pesan error
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Gagal mereset konfigurasi backbone",
                        alert_type='error'
                    ))
                return
            
            # Update UI dari konfigurasi default
            update_ui_from_config(ui_components, config)
            
            # Update info panel
            update_backbone_info(ui_components)
            
            # Sinkronisasi ke Google Drive jika diaktifkan
            try:
                env_manager = get_environment_manager()
                if env_manager.is_drive_mounted:
                    # Sinkronisasi ke Google Drive
                    logger.info("Menyinkronkan konfigurasi backbone ke Google Drive...")
                    sync_to_drive(None, ui_components)
            except Exception as e:
                logger.warning(f"Gagal menyinkronkan ke Google Drive: {str(e)}")
            
            logger.info("Konfigurasi backbone berhasil direset ke default")
            
            # Tampilkan pesan sukses
            with status_panel:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil direset ke default",
                    alert_type='success'
                ))
                
        except Exception as e:
            logger.error(f"Error saat reset konfigurasi backbone: {str(e)}")
            
            # Tampilkan pesan error
            with status_panel:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS.get('error', '❌')} Error saat reset konfigurasi backbone: {str(e)}",
                    alert_type='error'
                ))
