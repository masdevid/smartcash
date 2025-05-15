"""
File: smartcash/ui/training_config/backbone/handlers/button_handlers.py
Deskripsi: Handler untuk tombol pada UI pemilihan backbone model SmartCash
"""

from typing import Dict, Any, Callable
import ipywidgets as widgets
from IPython.display import clear_output, display

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert, create_status_indicator
from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.ui.training_config.backbone.handlers.config_handlers import (
    update_config_from_ui,
    update_ui_from_config,
    update_backbone_info
)
from smartcash.ui.training_config.backbone.handlers.drive_handlers import sync_to_drive

logger = get_logger(__name__)

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
        try:
            # Dapatkan ConfigManager
            config_manager = get_config_manager()
            
            # Update config dari UI
            config_to_save = update_config_from_ui(ui_components)
            
            # Simpan config ke file
            success = config_manager.save_module_config('model', config_to_save)
            
            if success:
                # Pastikan UI components teregistrasi untuk persistensi
                config_manager.register_ui_components('backbone', ui_components)
                
                # Tampilkan pesan sukses
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil disimpan",
                        alert_type='success'
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
                
                logger.info(f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil disimpan")
            else:
                # Tampilkan pesan error
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi backbone",
                        alert_type='error'
                    ))
                
                logger.error(f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi backbone")
        except Exception as e:
            # Tampilkan pesan error
            with status_panel:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi backbone: {str(e)}",
                    alert_type='error'
                ))
            
            logger.error(f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi backbone: {str(e)}")

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
        try:
            # Dapatkan ConfigManager
            config_manager = get_config_manager()
            
            # Dapatkan default config
            default_config = config_manager.get_module_config('model', {})
            
            # Update UI dari default config
            update_ui_from_config(ui_components, default_config)
            
            # Simpan default config ke file
            success = config_manager.save_module_config('model', default_config)
            
            if success:
                # Pastikan UI components teregistrasi untuk persistensi
                config_manager.register_ui_components('backbone', ui_components)
                
                # Tampilkan pesan sukses
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil direset ke default",
                        alert_type='success'
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
                
                logger.info(f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil direset ke default")
            else:
                # Tampilkan pesan error
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Gagal mereset konfigurasi backbone",
                        alert_type='error'
                    ))
                
                logger.error(f"{ICONS.get('error', '❌')} Gagal mereset konfigurasi backbone")
        except Exception as e:
            # Tampilkan pesan error
            with status_panel:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS.get('error', '❌')} Error saat mereset konfigurasi backbone: {str(e)}",
                    alert_type='error'
                ))
            
            logger.error(f"{ICONS.get('error', '❌')} Error saat mereset konfigurasi backbone: {str(e)}")
