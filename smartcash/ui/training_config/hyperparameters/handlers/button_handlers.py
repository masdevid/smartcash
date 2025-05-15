"""
File: smartcash/ui/training_config/hyperparameters/handlers/button_handlers.py
Deskripsi: Handler untuk tombol pada komponen UI hyperparameter
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert, create_status_indicator
from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.ui.training_config.hyperparameters.handlers.drive_handlers import sync_to_drive, sync_from_drive
from smartcash.ui.training_config.hyperparameters.handlers.config_handlers import update_config_from_ui, update_ui_from_config

logger = get_logger(__name__)

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
        # Dapatkan environment manager jika belum tersedia
        env = env or get_environment_manager()
        
        # Dapatkan config manager
        config_manager = get_config_manager()
        
        # Validasi config
        if config is None:
            config = config_manager.get_module_config('hyperparameters', {})
        
        # Pastikan ui_components memiliki referensi ke config
        ui_components['config'] = config
        
        # Handler untuk tombol save
        def on_save_click(b):
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator('info', f"{ICONS.get('info', 'ℹ️')} Menyimpan konfigurasi hyperparameter..."))
            
            try:
                # Update config dari UI
                updated_config = update_config_from_ui(ui_components, config.copy())
                
                # Simpan konfigurasi
                success = config_manager.save_module_config('hyperparameters', updated_config)
                
                # Pastikan UI components teregistrasi untuk persistensi
                try:
                    config_manager.register_ui_components('hyperparameters', ui_components)
                except Exception as persist_error:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat memastikan persistensi UI: {persist_error}")
                
                # Tampilkan pesan sukses atau warning
                with ui_components['status']:
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
                
                # Sinkronisasi ke Google Drive jika diaktifkan
                try:
                    if env.is_drive_mounted:
                        # Sinkronisasi ke Google Drive
                        logger.info(f"{ICONS.get('info', 'ℹ️')} Menyinkronkan konfigurasi hyperparameter ke Google Drive...")
                        sync_to_drive(None, ui_components)
                except Exception as e:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menyinkronkan ke Google Drive: {str(e)}")
                
                logger.info(f"{ICONS.get('success', '✅')} Konfigurasi hyperparameter berhasil disimpan")
            except Exception as e:
                with ui_components['status']:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi: {str(e)}",
                        alert_type='error'
                    ))
                
                logger.error(f"{ICONS.get('error', '❌')} Gagal menyimpan konfigurasi: {str(e)}")
        
        # Handler untuk tombol reset
        def on_reset_click(b):
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator('info', f"{ICONS.get('info', 'ℹ️')} Mereset konfigurasi hyperparameter..."))
            
            try:
                # Dapatkan default config
                default_config = config_manager.get_module_config('hyperparameters', {})
                
                # Update UI dari default config
                update_ui_from_config(ui_components, default_config)
                
                # Simpan default config
                success = config_manager.save_module_config('hyperparameters', default_config)
                
                # Pastikan UI components teregistrasi untuk persistensi
                try:
                    config_manager.register_ui_components('hyperparameters', ui_components)
                except Exception as persist_error:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat memastikan persistensi UI: {persist_error}")
                
                # Tampilkan pesan sukses atau warning
                with ui_components['status']:
                    clear_output(wait=True)
                    if success:
                        display(create_info_alert(
                            f"{ICONS.get('success', '✅')} Konfigurasi hyperparameter berhasil direset ke default",
                            alert_type='success'
                        ))
                    else:
                        display(create_info_alert(
                            f"{ICONS.get('warning', '⚠️')} Konfigurasi hyperparameter direset di UI tetapi mungkin tidak tersimpan ke file",
                            alert_type='warning'
                        ))
                
                # Update info panel jika ada
                if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                    ui_components['update_hyperparameters_info']()
                
                # Sinkronisasi ke Google Drive jika diaktifkan
                try:
                    if env.is_drive_mounted:
                        # Sinkronisasi ke Google Drive
                        logger.info(f"{ICONS.get('info', 'ℹ️')} Menyinkronkan konfigurasi hyperparameter ke Google Drive...")
                        sync_to_drive(None, ui_components)
                except Exception as e:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menyinkronkan ke Google Drive: {str(e)}")
                
                logger.info(f"{ICONS.get('success', '✅')} Konfigurasi hyperparameter berhasil direset ke default")
            except Exception as e:
                with ui_components['status']:
                    clear_output(wait=True)
                    display(create_info_alert(
                        f"{ICONS.get('error', '❌')} Gagal mereset konfigurasi: {str(e)}",
                        alert_type='error'
                    ))
                
                logger.error(f"{ICONS.get('error', '❌')} Gagal mereset konfigurasi: {str(e)}")
        
        # Handler untuk tombol sync dari drive
        def on_sync_from_drive_click(b):
            sync_from_drive(b, ui_components)
        
        # Handler untuk tombol sync ke drive
        def on_sync_to_drive_click(b):
            sync_to_drive(b, ui_components)
        
        # Pasang handler ke tombol
        if 'save_button' in ui_components:
            ui_components['save_button'].on_click(on_save_click)
        
        if 'reset_button' in ui_components:
            ui_components['reset_button'].on_click(on_reset_click)
        
        if 'sync_from_drive_button' in ui_components:
            ui_components['sync_from_drive_button'].on_click(on_sync_from_drive_click)
        
        if 'sync_to_drive_button' in ui_components:
            ui_components['sync_to_drive_button'].on_click(on_sync_to_drive_click)
        
        # Tambahkan handler ke ui_components
        ui_components.update({
            'on_save_click': on_save_click,
            'on_reset_click': on_reset_click,
            'on_sync_from_drive_click': on_sync_from_drive_click,
            'on_sync_to_drive_click': on_sync_to_drive_click
        })
        
        return ui_components
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat setup button handlers: {str(e)}")
        return ui_components
