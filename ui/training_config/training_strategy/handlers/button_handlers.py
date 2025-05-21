"""
File: smartcash/ui/training_config/training_strategy/handlers/button_handlers.py
Deskripsi: Handler untuk tombol-tombol di UI konfigurasi training strategy
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.ui.utils.constants import ICONS
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.ui.training_config.training_strategy.handlers.config_handlers import (
    update_config_from_ui,
    update_ui_from_config,
    get_default_config,
    update_training_strategy_info
)
from smartcash.ui.training_config.training_strategy.handlers.status_handlers import update_status_panel
from smartcash.ui.training_config.training_strategy.handlers.sync_logger import update_sync_status_only

logger = get_logger(__name__)

def get_default_base_dir():
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def on_save_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol Save.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    try:
        # Update status
        update_status_panel(ui_components, "Menyimpan konfigurasi strategi pelatihan...", "info")
        
        # Update config dari UI
        config = update_config_from_ui(ui_components)
        
        # Simpan config
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        saved_config = config_manager.save_module_config('training_strategy', config)
        
        # Update UI dari config yang disimpan untuk memastikan konsistensi
        update_ui_from_config(ui_components, config)
        
        # Update info panel
        update_training_strategy_info(ui_components)
        
        # Update status
        update_status_panel(ui_components, "Konfigurasi strategi pelatihan berhasil disimpan", "success")
        
        # Log untuk debugging
        logger.info(f"Konfigurasi strategi pelatihan berhasil disimpan dan UI diperbarui")
        
    except Exception as e:
        logger.error(f"Error saat menyimpan konfigurasi strategi pelatihan: {str(e)}")
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
        update_status_panel(ui_components, "Mereset konfigurasi strategi pelatihan...", "info")
        
        # Dapatkan konfigurasi default
        default_config = get_default_config()
        
        # Update UI dari konfigurasi default terlebih dahulu
        update_ui_from_config(ui_components, default_config)
        
        # Simpan konfigurasi default
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        saved_config = config_manager.save_module_config('training_strategy', default_config)
        
        # Update info panel
        update_training_strategy_info(ui_components)
        
        # Update status
        update_status_panel(ui_components, "Konfigurasi strategi pelatihan berhasil direset ke default", "success")
        
        logger.info("Konfigurasi strategi pelatihan berhasil direset ke default")
        
    except Exception as e:
        logger.error(f"Error saat mereset konfigurasi strategi pelatihan: {str(e)}")
        update_status_panel(ui_components, f"Error: {str(e)}", "error")

def setup_training_strategy_button_handlers(ui_components: Dict[str, Any], env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol-tombol di UI konfigurasi strategi pelatihan.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (opsional)
        config: Konfigurasi strategi pelatihan (opsional)
        
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
    
    # Add handler functions to ui_components for testing
    ui_components['on_save_click'] = lambda b: on_save_click(b, ui_components)
    ui_components['on_reset_click'] = lambda b: on_reset_click(b, ui_components)
    
    return ui_components
