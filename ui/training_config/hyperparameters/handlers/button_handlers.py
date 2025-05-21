"""
File: smartcash/ui/training_config/hyperparameters/handlers/button_handlers.py
Deskripsi: Handler untuk tombol-tombol di UI konfigurasi hyperparameters
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS
from smartcash.common.config import get_config_manager
from smartcash.ui.training_config.hyperparameters.handlers.default_config import get_default_hyperparameters_config
from smartcash.ui.training_config.hyperparameters.handlers.config_reader import update_config_from_ui
from smartcash.ui.training_config.hyperparameters.handlers.config_writer import update_ui_from_config
from smartcash.ui.training_config.hyperparameters.handlers.config_manager import (
    get_hyperparameters_config,
    save_hyperparameters_config,
    reset_hyperparameters_config,
    get_default_base_dir
)
from smartcash.ui.training_config.hyperparameters.handlers.drive_handlers import (
    sync_to_drive,
    sync_from_drive
)
from smartcash.ui.training_config.hyperparameters.handlers.status_handlers import (
    update_status_panel,
    show_success_status,
    show_error_status,
    show_info_status,
    show_warning_status
)

logger = get_logger(__name__)

def on_save_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol Save.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    try:
        # Update status
        show_info_status(ui_components, "Menyimpan konfigurasi hyperparameter...")
        
        # Update config dari UI
        config = update_config_from_ui(ui_components)
        
        # Simpan config
        save_success = save_hyperparameters_config(config)
        
        if not save_success:
            show_error_status(ui_components, "Gagal menyimpan konfigurasi hyperparameter")
            return
        
        # Update UI dari config yang disimpan untuk memastikan konsistensi
        saved_config = get_hyperparameters_config()
        update_ui_from_config(ui_components, saved_config)
        
        # Update info panel jika ada
        if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
            ui_components['update_hyperparameters_info'](ui_components)
        
        # Update status
        show_success_status(ui_components, "Konfigurasi hyperparameter berhasil disimpan")
        
        # Log untuk debugging
        logger.info(f"Konfigurasi hyperparameter berhasil disimpan dan UI diperbarui")
        
    except Exception as e:
        logger.error(f"Error saat menyimpan konfigurasi hyperparameters: {str(e)}")
        show_error_status(ui_components, f"Error: {str(e)}")

def on_reset_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol Reset.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    try:
        # Update status
        show_info_status(ui_components, "Mereset konfigurasi hyperparameter...")
        
        # Dapatkan konfigurasi default
        default_config = get_default_hyperparameters_config()
        
        # Update UI dari konfigurasi default terlebih dahulu
        update_ui_from_config(ui_components, default_config)
        
        # Simpan konfigurasi default
        save_success = save_hyperparameters_config(default_config)
        
        if not save_success:
            show_error_status(ui_components, "Gagal menyimpan konfigurasi default hyperparameter")
            return
        
        # Update info panel jika ada
        if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
            ui_components['update_hyperparameters_info'](ui_components)
        
        # Update status
        show_success_status(ui_components, "Konfigurasi hyperparameter berhasil direset ke default")
        
    except Exception as e:
        logger.error(f"Error saat mereset konfigurasi hyperparameters: {str(e)}")
        show_error_status(ui_components, f"Error: {str(e)}")

def on_sync_to_drive_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol Sync to Drive.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    try:
        # Update status
        show_info_status(ui_components, "Menyinkronkan konfigurasi hyperparameters ke Google Drive...")
        
        # Update config dari UI
        config = update_config_from_ui(ui_components)
        
        # Simpan config
        save_success = save_hyperparameters_config(config)
        
        if not save_success:
            show_error_status(ui_components, "Gagal menyimpan konfigurasi sebelum sinkronisasi")
            return
        
        # Sinkronisasi ke Drive
        success, message = sync_to_drive(button, ui_components)
        
        if success:
            # Update status
            show_success_status(ui_components, message)
        else:
            # Update status
            show_error_status(ui_components, message)
        
    except Exception as e:
        logger.error(f"Error saat sinkronisasi ke Google Drive: {str(e)}")
        show_error_status(ui_components, f"Error: {str(e)}")

def on_sync_from_drive_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol Sync from Drive.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    try:
        # Update status
        show_info_status(ui_components, "Menyinkronkan konfigurasi hyperparameters dari Google Drive...")
        
        # Sinkronisasi dari Drive
        success, message, drive_config = sync_from_drive(button, ui_components)
        
        if success and drive_config:
            # Update UI dari konfigurasi yang disinkronkan
            update_ui_from_config(ui_components, drive_config)
            
            # Update info panel jika ada
            if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                ui_components['update_hyperparameters_info'](ui_components)
            
            # Update status
            show_success_status(ui_components, message)
        else:
            # Update status
            show_error_status(ui_components, message)
        
    except Exception as e:
        logger.error(f"Error saat sinkronisasi dari Google Drive: {str(e)}")
        show_error_status(ui_components, f"Error: {str(e)}")

def setup_hyperparameters_button_handlers(ui_components: Dict[str, Any], env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol-tombol di UI konfigurasi hyperparameters.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (opsional)
        config: Konfigurasi hyperparameters (opsional)
        
    Returns:
        Dictionary komponen UI yang telah diupdate dengan handler
    """
    # Handler untuk tombol Save
    if 'save_button' in ui_components:
        ui_components['save_button'].on_click(
            lambda b: on_save_click(b, ui_components)
        )
    
    # Handler untuk tombol Reset
    if 'reset_button' in ui_components:
        ui_components['reset_button'].on_click(
            lambda b: on_reset_click(b, ui_components)
        )
    
    # Handler untuk tombol Sync to Drive (jika ada)
    if 'sync_to_drive_button' in ui_components:
        ui_components['sync_to_drive_button'].on_click(
            lambda b: on_sync_to_drive_click(b, ui_components)
        )
    
    # Handler untuk tombol Sync from Drive (jika ada)
    if 'sync_from_drive_button' in ui_components:
        ui_components['sync_from_drive_button'].on_click(
            lambda b: on_sync_from_drive_click(b, ui_components)
        )
    
    # Add handler functions to ui_components for testing
    ui_components['on_save_click'] = lambda b: on_save_click(b, ui_components)
    ui_components['on_reset_click'] = lambda b: on_reset_click(b, ui_components)
    ui_components['on_sync_to_drive_click'] = lambda b: on_sync_to_drive_click(b, ui_components)
    ui_components['on_sync_from_drive_click'] = lambda b: on_sync_from_drive_click(b, ui_components)
    
    return ui_components