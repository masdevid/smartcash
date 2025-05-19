"""
File: smartcash/ui/training_config/training_strategy/handlers/drive_handlers.py
Deskripsi: Handler untuk sinkronisasi konfigurasi strategi pelatihan dengan Google Drive
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, clear_output
import os
import shutil
from pathlib import Path

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert, create_status_indicator
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.common.io import save_yaml
from smartcash.ui.training_config.training_strategy.handlers.config_handlers import update_ui_from_config, update_config_from_ui

logger = get_logger(__name__)

def get_default_base_dir():
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def sync_to_drive(b=None, ui_components: Dict[str, Any] = None) -> None:
    """
    Sinkronisasi konfigurasi strategi pelatihan ke Google Drive.
    
    Args:
        b: Button event (opsional)
        ui_components: Dictionary komponen UI
    """
    if ui_components is None:
        logger.error(f"{ICONS.get('error', '❌')} UI components tidak tersedia")
        return
    
    # Dapatkan environment manager
    env = get_environment_manager(base_dir=get_default_base_dir())
    
    # Dapatkan config manager
    config_manager = get_config_manager(base_dir=get_default_base_dir())
    
    # Tampilkan status
    with ui_components['status']:
        clear_output(wait=True)
        display(create_status_indicator('info', f"{ICONS.get('info', 'ℹ️')} Menyinkronkan konfigurasi strategi pelatihan ke Google Drive..."))
    
    try:
        # Pastikan Google Drive sudah di-mount
        if not env.is_drive_mounted:
            with ui_components['status']:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS.get('error', '❌')} Google Drive belum di-mount. Silakan mount Google Drive terlebih dahulu.",
                    alert_type='error'
                ))
            return
        
        # Dapatkan path Google Drive
        drive_path = env.drive_path
        
        # Buat direktori konfigurasi di Google Drive jika belum ada
        drive_config_dir = os.path.join(drive_path, 'smartcash', 'configs')
        os.makedirs(drive_config_dir, exist_ok=True)
        
        # Dapatkan konfigurasi dari UI
        config = update_config_from_ui(ui_components)
        
        # Simpan konfigurasi ke file di Google Drive
        drive_config_file = os.path.join(drive_config_dir, 'training_config.yaml')
        # Gunakan save_config dengan parameter yang benar
        success = config_manager.save_config(drive_config_file, create_dirs=True)
        
        # Simpan konfigurasi ke file di Google Drive menggunakan save_yaml
        try:
            save_yaml(config, drive_config_file)
            success = True
        except Exception as e:
            logger.error(f"{ICONS.get('error', '❌')} Error saat menyimpan ke file: {str(e)}")
            success = False
        
        # Tampilkan status
        with ui_components['status']:
            clear_output(wait=True)
            if success:
                display(create_info_alert(
                    f"{ICONS.get('success', '✅')} Konfigurasi strategi pelatihan berhasil disinkronkan ke Google Drive",
                    alert_type='success'
                ))
            else:
                display(create_info_alert(
                    f"{ICONS.get('warning', '⚠️')} Konfigurasi strategi pelatihan mungkin tidak tersinkronkan dengan benar",
                    alert_type='warning'
                ))
        
        logger.info(f"{ICONS.get('success', '✅')} Konfigurasi strategi pelatihan berhasil disinkronkan ke Google Drive")
    except Exception as e:
        with ui_components['status']:
            clear_output(wait=True)
            display(create_info_alert(
                f"{ICONS.get('error', '❌')} Gagal menyinkronkan konfigurasi ke Google Drive: {str(e)}",
                alert_type='error'
            ))
        
        logger.error(f"{ICONS.get('error', '❌')} Gagal menyinkronkan konfigurasi ke Google Drive: {str(e)}")

def sync_from_drive(b=None, ui_components: Dict[str, Any] = None) -> None:
    """
    Sinkronisasi konfigurasi strategi pelatihan dari Google Drive.
    
    Args:
        b: Button event (opsional)
        ui_components: Dictionary komponen UI
    """
    if ui_components is None:
        logger.error(f"{ICONS.get('error', '❌')} UI components tidak tersedia")
        return
    
    # Dapatkan environment manager
    env = get_environment_manager(base_dir=get_default_base_dir())
    
    # Dapatkan config manager
    config_manager = get_config_manager(base_dir=get_default_base_dir())
    
    # Tampilkan status
    with ui_components['status']:
        clear_output(wait=True)
        display(create_status_indicator('info', f"{ICONS.get('info', 'ℹ️')} Menyinkronkan konfigurasi strategi pelatihan dari Google Drive..."))
    
    try:
        # Pastikan Google Drive sudah di-mount
        if not env.is_drive_mounted:
            with ui_components['status']:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS.get('error', '❌')} Google Drive belum di-mount. Silakan mount Google Drive terlebih dahulu.",
                    alert_type='error'
                ))
            return
        
        # Dapatkan path Google Drive
        drive_path = env.drive_path
        
        # Dapatkan path file konfigurasi di Google Drive
        drive_config_file = os.path.join(drive_path, 'smartcash', 'configs', 'training_config.yaml')
        
        # Pastikan file konfigurasi ada di Google Drive
        if not os.path.exists(drive_config_file):
            with ui_components['status']:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS.get('error', '❌')} File konfigurasi tidak ditemukan di Google Drive.",
                    alert_type='error'
                ))
            return
        
        # Load konfigurasi dari Google Drive
        config = config_manager.load_config(drive_config_file)
        
        # Simpan konfigurasi ke file lokal
        success = config_manager.save_module_config('training_strategy', config)
        
        # Update UI dari konfigurasi yang diload
        update_ui_from_config(ui_components, config)
        
        # Tampilkan status
        with ui_components['status']:
            clear_output(wait=True)
            if success:
                display(create_info_alert(
                    f"{ICONS.get('success', '✅')} Konfigurasi strategi pelatihan berhasil disinkronkan dari Google Drive",
                    alert_type='success'
                ))
            else:
                display(create_info_alert(
                    f"{ICONS.get('warning', '⚠️')} Konfigurasi strategi pelatihan mungkin tidak tersinkronkan dengan benar",
                    alert_type='warning'
                ))
        
        logger.info(f"{ICONS.get('success', '✅')} Konfigurasi strategi pelatihan berhasil disinkronkan dari Google Drive")
    except Exception as e:
        with ui_components['status']:
            clear_output(wait=True)
            display(create_info_alert(
                f"{ICONS.get('error', '❌')} Gagal menyinkronkan konfigurasi dari Google Drive: {str(e)}",
                alert_type='error'
            ))
        
        logger.error(f"{ICONS.get('error', '❌')} Gagal menyinkronkan konfigurasi dari Google Drive: {str(e)}")
