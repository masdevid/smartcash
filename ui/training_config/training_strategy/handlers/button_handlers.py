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
from smartcash.ui.training_config.training_strategy.handlers.config_loader import get_default_base_dir
from smartcash.ui.training_config.training_strategy.handlers.default_config import get_default_config
from smartcash.ui.training_config.training_strategy.handlers.ui_updater import update_ui_from_config
from smartcash.ui.training_config.training_strategy.handlers.config_extractor import update_config_from_ui
from smartcash.ui.training_config.training_strategy.handlers.info_updater import update_training_strategy_info
from smartcash.ui.training_config.training_strategy.handlers.status_handlers import update_status_panel
from smartcash.ui.training_config.training_strategy.handlers.sync_logger import update_sync_status_only

logger = get_logger(__name__)

def on_save_click(button: Optional[widgets.Button], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol Save.
    
    Args:
        button: Tombol yang diklik (bisa None jika dipanggil secara programatis)
        ui_components: Dictionary komponen UI
    """
    try:
        logger.info(f"{ICONS.get('info', 'ℹ️')} Memproses tombol simpan...")
        
        # Disable tombol selama proses penyimpanan
        if button is not None:
            button.disabled = True
        
        # Update status
        update_status_panel(ui_components, "Menyimpan konfigurasi strategi pelatihan...", "info")
        
        # Update config dari UI - ambil konfigurasi terbaru dari UI components
        config = update_config_from_ui(ui_components)
        
        # Simpan config ke file
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        result = config_manager.save_module_config('training_strategy', config)
        
        if not result:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menyimpan konfigurasi strategi pelatihan")
            update_status_panel(ui_components, "Gagal menyimpan konfigurasi strategi pelatihan", "warning")
            if button is not None:
                button.disabled = False
            return
        
        # Pastikan UI dan config tetap sinkron
        update_ui_from_config(ui_components, config)
        
        # Update info panel dengan pesan sukses
        if 'update_training_strategy_info' in ui_components and callable(ui_components['update_training_strategy_info']):
            ui_components['update_training_strategy_info'](ui_components)
        else:
            update_training_strategy_info(ui_components, "Konfigurasi berhasil disimpan")
        
        # Update status dengan pesan sukses
        update_status_panel(ui_components, "Konfigurasi strategi pelatihan berhasil disimpan", "success")
        
        # Coba sinkronkan ke Google Drive jika Colab
        try:
            if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
                from smartcash.ui.training_config.training_strategy.handlers.drive_handlers import sync_to_drive
                sync_to_drive(ui_components=ui_components)
        except Exception as drive_error:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Sinkronisasi ke Drive gagal: {str(drive_error)}")
        
        logger.info(f"{ICONS.get('success', '✅')} Konfigurasi strategi pelatihan berhasil disimpan")
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi: {str(e)}")
        update_status_panel(ui_components, f"Error: {str(e)}", "error")
    finally:
        # Enable kembali tombol
        if button is not None:
            button.disabled = False

def on_reset_click(button: Optional[widgets.Button], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol Reset.
    
    Args:
        button: Tombol yang diklik (bisa None jika dipanggil secara programatis)
        ui_components: Dictionary komponen UI
    """
    try:
        logger.info(f"{ICONS.get('info', 'ℹ️')} Memproses tombol reset...")
        
        # Disable tombol selama proses reset
        if button is not None:
            button.disabled = True
        
        # Update status
        update_status_panel(ui_components, "Mereset konfigurasi strategi pelatihan...", "info")
        
        # Dapatkan konfigurasi default
        default_config = get_default_config()
        
        # Update UI dari konfigurasi default
        update_ui_from_config(ui_components, default_config)
        
        # Simpan konfigurasi default ke file
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        result = config_manager.save_module_config('training_strategy', default_config)
        
        if not result:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menyimpan konfigurasi default")
            update_status_panel(ui_components, "Gagal menyimpan konfigurasi default", "warning")
            if button is not None:
                button.disabled = False
            return
        
        # Update info panel dengan pesan sukses
        if 'update_training_strategy_info' in ui_components and callable(ui_components['update_training_strategy_info']):
            ui_components['update_training_strategy_info'](ui_components)
        else:
            update_training_strategy_info(ui_components, "Konfigurasi berhasil direset ke default")
        
        # Update status
        update_status_panel(ui_components, "Konfigurasi strategi pelatihan berhasil direset ke default", "success")
        
        # Coba sinkronkan ke Google Drive jika Colab
        try:
            if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
                from smartcash.ui.training_config.training_strategy.handlers.drive_handlers import sync_to_drive
                sync_to_drive(ui_components=ui_components)
        except Exception as drive_error:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Sinkronisasi ke Drive gagal: {str(drive_error)}")
        
        logger.info(f"{ICONS.get('success', '✅')} Konfigurasi strategi pelatihan berhasil direset ke default")
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat mereset konfigurasi: {str(e)}")
        update_status_panel(ui_components, f"Error: {str(e)}", "error")
    finally:
        # Enable kembali tombol
        if button is not None:
            button.disabled = False

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
    logger.info(f"{ICONS.get('info', 'ℹ️')} Setting up button handlers untuk strategi pelatihan")
    
    # Handler untuk tombol Save
    if 'save_button' in ui_components:
        ui_components['save_button'].on_click(
            lambda b: on_save_click(b, ui_components)
        )
        logger.info(f"{ICONS.get('success', '✅')} Handler untuk save_button berhasil dipasang")
    else:
        logger.warning(f"{ICONS.get('warning', '⚠️')} save_button tidak ditemukan di ui_components")
    
    # Handler untuk tombol Reset
    if 'reset_button' in ui_components:
        ui_components['reset_button'].on_click(
            lambda b: on_reset_click(b, ui_components)
        )
        logger.info(f"{ICONS.get('success', '✅')} Handler untuk reset_button berhasil dipasang")
    else:
        logger.warning(f"{ICONS.get('warning', '⚠️')} reset_button tidak ditemukan di ui_components")
    
    # Add handler functions to ui_components for testing and easier access
    ui_components['on_save_click'] = lambda b=None: on_save_click(b or ui_components.get('save_button'), ui_components)
    ui_components['on_reset_click'] = lambda b=None: on_reset_click(b or ui_components.get('reset_button'), ui_components)
    
    return ui_components