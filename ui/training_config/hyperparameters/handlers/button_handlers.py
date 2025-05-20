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
from smartcash.ui.training_config.hyperparameters.default_config import get_default_hyperparameters_config
from smartcash.ui.training_config.hyperparameters.handlers.config_handlers import (
    update_ui_from_config,
    update_config_from_ui,
    save_config,
    get_hyperparameters_config,
    update_hyperparameters_info
)
from smartcash.ui.training_config.hyperparameters.handlers.drive_handlers import (
    sync_to_drive,
    sync_from_drive
)
from smartcash.ui.training_config.hyperparameters.handlers.status_handlers import (
    update_status_panel
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
        update_status_panel(ui_components, "Menyimpan konfigurasi hyperparameter...", "info")
        
        # Update config dari UI
        config = update_config_from_ui(ui_components)
        
        # Simpan config
        saved_config = save_config(config, ui_components)
        
        # Update UI dari config yang disimpan untuk memastikan konsistensi
        update_ui_from_config(ui_components, saved_config)
        
        # Update info panel
        update_hyperparameters_info(ui_components)
        
        # Update status
        update_status_panel(ui_components, "Konfigurasi hyperparameter berhasil disimpan", "success")
        
    except Exception as e:
        logger.error(f"Error saat menyimpan konfigurasi hyperparameters: {str(e)}")
        update_status_panel(ui_components, f"Error: {str(e)}", "error")

def on_reset_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol Reset.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    try:
        # Update status
        update_status_panel(ui_components, "Mereset konfigurasi hyperparameter...", "info")
        
        # Dapatkan konfigurasi default
        default_config = get_default_hyperparameters_config()
        
        # Simpan konfigurasi default
        saved_config = save_config(default_config, ui_components)
        
        # Update UI dari konfigurasi default
        update_ui_from_config(ui_components, saved_config)
        
        # Update info panel
        update_hyperparameters_info(ui_components)
        
        # Update status
        update_status_panel(ui_components, "Konfigurasi hyperparameter berhasil direset ke default", "success")
        
    except Exception as e:
        logger.error(f"Error saat mereset konfigurasi hyperparameters: {str(e)}")
        update_status_panel(ui_components, f"Error: {str(e)}", "error")

def on_sync_to_drive_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol Sync to Drive.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    try:
        # Update status
        update_status_panel(ui_components, "Menyinkronkan konfigurasi hyperparameters ke Google Drive...", "info")
        
        # Update config dari UI
        config = update_config_from_ui(ui_components)
        
        # Simpan config
        saved_config = save_config(config, ui_components)
        
        # Sinkronisasi ke Drive
        success, message = sync_to_drive(button, ui_components)
        
        if success:
            # Update status
            update_status_panel(ui_components, message, "success")
        else:
            # Update status
            update_status_panel(ui_components, message, "error")
        
    except Exception as e:
        logger.error(f"Error saat sinkronisasi ke Google Drive: {str(e)}")
        update_status_panel(ui_components, f"Error: {str(e)}", "error")

def on_sync_from_drive_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol Sync from Drive.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    try:
        # Update status
        update_status_panel(ui_components, "Menyinkronkan konfigurasi hyperparameters dari Google Drive...", "info")
        
        # Sinkronisasi dari Drive
        success, message, drive_config = sync_from_drive(button, ui_components)
        
        if success and drive_config:
            # Update UI dari konfigurasi yang disinkronkan
            update_ui_from_config(ui_components, drive_config)
            
            # Update info panel
            update_hyperparameters_info(ui_components)
            
            # Update status
            update_status_panel(ui_components, message, "success")
        else:
            # Update status
            update_status_panel(ui_components, message, "error")
        
    except Exception as e:
        logger.error(f"Error saat sinkronisasi dari Google Drive: {str(e)}")
        update_status_panel(ui_components, f"Error: {str(e)}", "error")

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
    
    # Handler untuk tombol Sync to Drive
    if 'sync_to_drive_button' in ui_components:
        ui_components['sync_to_drive_button'].on_click(
            lambda b: on_sync_to_drive_click(b, ui_components)
        )
    
    # Handler untuk tombol Sync from Drive
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