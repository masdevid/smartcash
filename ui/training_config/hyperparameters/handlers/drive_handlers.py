"""
File: smartcash/ui/training_config/hyperparameters/handlers/drive_handlers.py
Deskripsi: Handler untuk sinkronisasi konfigurasi hyperparameter dengan Google Drive
"""

from typing import Dict, Any, Optional
import os
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert, create_status_indicator
from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.ui.training_config.hyperparameters.handlers.config_handlers import update_ui_from_config

logger = get_logger(__name__)

def sync_to_drive(button: Optional[widgets.Button], ui_components: Dict[str, Any]) -> None:
    """
    Sinkronisasi konfigurasi hyperparameter ke Google Drive.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary berisi komponen UI
    """
    status_panel = ui_components.get('status')
    if not status_panel:
        logger.error(f"{ICONS.get('error', '❌')} Status panel tidak ditemukan")
        return
    
    with status_panel:
        clear_output(wait=True)
        display(create_status_indicator('info', f"{ICONS.get('info', 'ℹ️')} Menyinkronkan konfigurasi hyperparameter ke Google Drive..."))
        try:
            # Dapatkan environment manager
            env_manager = get_environment_manager()
            
            # Cek apakah drive diaktifkan
            if not env_manager.is_drive_mounted:
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi.",
                        alert_type='error'
                    ))
                return
            
            # Dapatkan ConfigManager
            config_manager = get_config_manager()
            
            # Dapatkan konfigurasi
            config = config_manager.get_module_config('hyperparameters')
            
            # Dapatkan path file konfigurasi di drive
            drive_config_path = os.path.join(
                env_manager.drive_path,
                'configs',
                'hyperparameters_config.yaml'
            )
            
            # Pastikan direktori ada
            os.makedirs(os.path.dirname(drive_config_path), exist_ok=True)
            
            # Simpan konfigurasi ke drive
            try:
                config_manager.save_config(drive_config_path, config, create_dirs=True)
                success = True
            except Exception as e:
                logger.error(f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi ke drive: {str(e)}")
                success = False
            
            if success:
                # Tampilkan pesan sukses
                with status_panel:
                    display(create_info_alert(
                        f"{ICONS.get('success', '✅')} Konfigurasi hyperparameter berhasil disinkronkan ke Google Drive",
                        alert_type='success'
                    ))
                
                logger.info(f"{ICONS.get('success', '✅')} Konfigurasi hyperparameter berhasil disinkronkan ke Google Drive")
            else:
                # Tampilkan pesan error
                with status_panel:
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Gagal menyinkronkan konfigurasi hyperparameter ke Google Drive",
                        alert_type='error'
                    ))
                
                logger.error(f"{ICONS.get('error', '❌')} Gagal menyinkronkan konfigurasi hyperparameter ke Google Drive")
        except Exception as e:
            # Tampilkan pesan error
            with status_panel:
                display(create_info_alert(
                    f"{ICONS.get('error', '❌')} Error saat menyinkronkan konfigurasi hyperparameter ke Google Drive: {str(e)}",
                    alert_type='error'
                ))
            
            logger.error(f"{ICONS.get('error', '❌')} Error saat menyinkronkan konfigurasi hyperparameter ke Google Drive: {str(e)}")

def sync_from_drive(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Sinkronisasi konfigurasi hyperparameter dari Google Drive.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary berisi komponen UI
    """
    status_panel = ui_components.get('status')
    if not status_panel:
        logger.error(f"{ICONS.get('error', '❌')} Status panel tidak ditemukan")
        return
    
    with status_panel:
        clear_output(wait=True)
        display(create_status_indicator('info', f"{ICONS.get('info', 'ℹ️')} Menyinkronkan konfigurasi hyperparameter dari Google Drive..."))
        try:
            # Dapatkan environment manager
            env_manager = get_environment_manager()
            
            # Cek apakah drive diaktifkan
            if not env_manager.is_drive_mounted:
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi.",
                        alert_type='error'
                    ))
                return
            
            # Dapatkan ConfigManager
            config_manager = get_config_manager()
            
            # Dapatkan path file konfigurasi di drive
            drive_config_path = os.path.join(
                env_manager.drive_path,
                'configs',
                'hyperparameters_config.yaml'
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
            try:
                from smartcash.common.io import load_yaml
                drive_config = load_yaml(drive_config_path, {})
            except Exception as e:
                logger.error(f"{ICONS.get('error', '❌')} Error saat memuat konfigurasi dari drive: {str(e)}")
                drive_config = None
            
            if drive_config:
                # Simpan konfigurasi ke lokal
                success = config_manager.save_module_config('hyperparameters', drive_config)
                
                if success:
                    # Update UI dari konfigurasi
                    update_ui_from_config(ui_components, drive_config)
                    
                    # Tampilkan pesan sukses
                    with status_panel:
                        display(create_info_alert(
                            f"{ICONS.get('success', '✅')} Konfigurasi hyperparameter berhasil disinkronkan dari Google Drive",
                            alert_type='success'
                        ))
                    
                    logger.info(f"{ICONS.get('success', '✅')} Konfigurasi hyperparameter berhasil disinkronkan dari Google Drive")
                else:
                    # Tampilkan pesan error
                    with status_panel:
                        display(create_info_alert(
                            f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi hyperparameter dari Google Drive",
                            alert_type='error'
                        ))
                    
                    logger.error(f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi hyperparameter dari Google Drive")
            else:
                # Tampilkan pesan error
                with status_panel:
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Gagal memuat konfigurasi hyperparameter dari Google Drive",
                        alert_type='error'
                    ))
                
                logger.error(f"{ICONS.get('error', '❌')} Gagal memuat konfigurasi hyperparameter dari Google Drive")
        except Exception as e:
            # Tampilkan pesan error
            with status_panel:
                display(create_info_alert(
                    f"{ICONS.get('error', '❌')} Error saat menyinkronkan konfigurasi hyperparameter dari Google Drive: {str(e)}",
                    alert_type='error'
                ))
            
            logger.error(f"{ICONS.get('error', '❌')} Error saat menyinkronkan konfigurasi hyperparameter dari Google Drive: {str(e)}")
