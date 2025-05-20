"""
File: smartcash/ui/training_config/hyperparameters/handlers/button_handlers.py
Deskripsi: Handler untuk tombol pada komponen UI hyperparameter
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, clear_output
from pathlib import Path
import os

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert, create_status_indicator
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger, LogLevel
from smartcash.common.environment import get_environment_manager
from smartcash.ui.training_config.hyperparameters.handlers.drive_handlers import sync_to_drive, sync_from_drive
from smartcash.ui.training_config.hyperparameters.handlers.config_handlers import update_config_from_ui, update_ui_from_config

# Setup logger dengan level CRITICAL untuk mengurangi log
logger = get_logger(__name__)
logger.set_level(LogLevel.CRITICAL)

def get_default_base_dir():
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def setup_hyperparameters_button_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol pada komponen UI hyperparameter.
    
    Args:
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI dengan handler terpasang
    """
    try:
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        
        # Handler untuk tombol save
        def on_save_click(b):
            status_panel = ui_components.get('status_panel', ui_components.get('status'))
            with status_panel:
                clear_output(wait=True)
                display(create_status_indicator('info', f"{ICONS.get('info', 'ℹ️')} Menyimpan konfigurasi hyperparameter..."))
            
            try:
                # Update config dari UI
                updated_config = update_config_from_ui(ui_components)
                
                # Simpan konfigurasi
                success = config_manager.save_module_config('hyperparameters', updated_config)
                
                # Pastikan UI components teregistrasi untuk persistensi
                try:
                    config_manager.register_ui_components('hyperparameters', ui_components)
                except Exception as persist_error:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat memastikan persistensi UI: {persist_error}")
                
                # Sinkronkan dengan drive
                try:
                    config_manager.sync_config_with_drive('hyperparameters')
                    logger.info("✅ Konfigurasi berhasil disinkronkan dengan drive")
                except Exception as sync_error:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat sinkronisasi dengan drive: {str(sync_error)}")
                
                # Tampilkan pesan sukses atau warning
                status_panel = ui_components.get('status_panel', ui_components.get('status'))
                with status_panel:
                    clear_output(wait=True)
                    if success:
                        display(create_info_alert(
                            f"{ICONS.get('success', '✅')} Konfigurasi hyperparameter berhasil disimpan",
                            alert_type='success'
                        ))
                    else:
                        display(create_info_alert(
                            f"{ICONS.get('warning', '⚠️')} Konfigurasi hyperparameter mungkin tidak tersimpan ke file",
                            alert_type='warning'
                        ))
                
                # Update info panel jika ada
                if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                    ui_components['update_hyperparameters_info']()
                
            except Exception as e:
                logger.error(f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi: {str(e)}")
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi: {str(e)}",
                        alert_type='error'
                    ))
        
        # Handler untuk tombol reset
        def on_reset_click(b):
            status_panel = ui_components.get('status_panel', ui_components.get('status'))
            with status_panel:
                clear_output(wait=True)
                display(create_status_indicator('info', f"{ICONS.get('info', 'ℹ️')} Mereset konfigurasi hyperparameter..."))
            
            try:
                # Reset ke konfigurasi default
                default_config = get_default_hyperparameters_config()
                update_ui_from_config(ui_components, default_config)
                
                # Simpan konfigurasi default
                success = config_manager.save_module_config('hyperparameters', default_config)
                
                # Sinkronkan dengan drive
                try:
                    config_manager.sync_config_with_drive('hyperparameters')
                    logger.info("✅ Konfigurasi default berhasil disinkronkan dengan drive")
                except Exception as sync_error:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat sinkronisasi dengan drive: {str(sync_error)}")
                
                # Tampilkan pesan sukses
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('success', '✅')} Konfigurasi hyperparameter berhasil direset ke default",
                        alert_type='success'
                    ))
                
                # Update info panel jika ada
                if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                    ui_components['update_hyperparameters_info']()
                
            except Exception as e:
                logger.error(f"{ICONS.get('error', '❌')} Error saat mereset konfigurasi: {str(e)}")
                with status_panel:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Error saat mereset konfigurasi: {str(e)}",
                        alert_type='error'
                    ))
        
        # Pasang handler ke tombol
        if 'save_button' in ui_components:
            ui_components['save_button'].on_click(on_save_click)
        
        if 'reset_button' in ui_components:
            ui_components['reset_button'].on_click(on_reset_click)
        
        # Tambahkan handler ke ui_components
        ui_components.update({
            'on_save_click': on_save_click,
            'on_reset_click': on_reset_click
        })
        
        return ui_components
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat setup button handlers: {str(e)}")
        return ui_components
