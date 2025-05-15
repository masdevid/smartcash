"""
File: smartcash/ui/training_config/backbone/handlers/drive_handlers.py
Deskripsi: Handler untuk sinkronisasi konfigurasi backbone dengan Google Drive
"""

from typing import Dict, Any, Optional
import os
import ipywidgets as widgets
from IPython.display import clear_output, display

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert, create_status_indicator
from smartcash.common.config.manager import ConfigManager
from smartcash.common.logger import get_logger
from smartcash.common.environment import EnvironmentManager
from smartcash.ui.training_config.backbone.handlers.config_handlers import update_ui_from_config

logger = get_logger(__name__)

def sync_to_drive(button: Optional[widgets.Button], ui_components: Dict[str, Any]) -> None:
    """
    Sinkronisasi konfigurasi backbone ke Google Drive.
    
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
            # Dapatkan environment manager
            env_manager = EnvironmentManager.get_instance()
            
            # Cek apakah drive diaktifkan
            if not env_manager.is_drive_enabled():
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi.",
                        alert_type='error'
                    ))
                return
            
            # Dapatkan ConfigManager
            config_manager = ConfigManager.get_instance()
            
            # Dapatkan konfigurasi
            config = config_manager.get_module_config('model')
            
            # Dapatkan path file konfigurasi di drive
            drive_config_path = os.path.join(
                env_manager.get_drive_path(),
                'configs',
                'model_config.yaml'
            )
            
            # Pastikan direktori ada
            os.makedirs(os.path.dirname(drive_config_path), exist_ok=True)
            
            # Simpan konfigurasi ke drive
            success = config_manager.save_config_to_file(config, drive_config_path)
            
            if success:
                # Tampilkan pesan sukses
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil disinkronkan ke Google Drive",
                        alert_type='success'
                    ))
                
                logger.info(f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil disinkronkan ke Google Drive")
            else:
                # Tampilkan pesan error
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Gagal menyinkronkan konfigurasi backbone ke Google Drive",
                        alert_type='error'
                    ))
                
                logger.error(f"{ICONS.get('error', '❌')} Gagal menyinkronkan konfigurasi backbone ke Google Drive")
        except Exception as e:
            # Tampilkan pesan error
            with status_panel:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS.get('error', '❌')} Error saat menyinkronkan konfigurasi backbone ke Google Drive: {str(e)}",
                    alert_type='error'
                ))
            
            logger.error(f"{ICONS.get('error', '❌')} Error saat menyinkronkan konfigurasi backbone ke Google Drive: {str(e)}")

def sync_from_drive(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Sinkronisasi konfigurasi backbone dari Google Drive.
    
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
            # Dapatkan environment manager
            env_manager = EnvironmentManager.get_instance()
            
            # Cek apakah drive diaktifkan
            if not env_manager.is_drive_enabled():
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi.",
                        alert_type='error'
                    ))
                return
            
            # Dapatkan ConfigManager
            config_manager = ConfigManager.get_instance()
            
            # Dapatkan path file konfigurasi di drive
            drive_config_path = os.path.join(
                env_manager.get_drive_path(),
                'configs',
                'model_config.yaml'
            )
            
            # Cek apakah file konfigurasi ada di drive
            if not os.path.exists(drive_config_path):
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} File konfigurasi tidak ditemukan di Google Drive",
                        alert_type='error'
                    ))
                return
            
            # Load konfigurasi dari drive
            drive_config = config_manager.load_config_from_file(drive_config_path)
            
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
                    
                    logger.info(f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil disinkronkan dari Google Drive")
                else:
                    # Tampilkan pesan error
                    with status_panel:
                        clear_output(wait=True)
                        display(create_info_alert(
                            f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi backbone dari Google Drive",
                            alert_type='error'
                        ))
                    
                    logger.error(f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi backbone dari Google Drive")
            else:
                # Tampilkan pesan error
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Gagal memuat konfigurasi backbone dari Google Drive",
                        alert_type='error'
                    ))
                
                logger.error(f"{ICONS.get('error', '❌')} Gagal memuat konfigurasi backbone dari Google Drive")
        except Exception as e:
            # Tampilkan pesan error
            with status_panel:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS.get('error', '❌')} Error saat menyinkronkan konfigurasi backbone dari Google Drive: {str(e)}",
                    alert_type='error'
                ))
            
            logger.error(f"{ICONS.get('error', '❌')} Error saat menyinkronkan konfigurasi backbone dari Google Drive: {str(e)}")
