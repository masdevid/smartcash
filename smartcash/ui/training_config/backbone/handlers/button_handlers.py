"""
File: smartcash/ui/training_config/backbone/handlers/button_handlers.py
Deskripsi: Handler untuk tombol pada UI pemilihan backbone model SmartCash
"""

from typing import Dict, Any
import ipywidgets as widgets
import os
from pathlib import Path

from smartcash.common.logger import get_logger, LogLevel
from smartcash.ui.training_config.backbone.handlers.config_handlers import (
    update_config_from_ui, 
    update_ui_from_config,
    update_backbone_info,
    get_default_backbone_config
)
from smartcash.ui.training_config.backbone.handlers.save_handlers import save_backbone_config
from smartcash.ui.training_config.backbone.handlers.status_handlers import update_status_panel

# Setup logger dengan level CRITICAL untuk mengurangi log
logger = get_logger(__name__)
logger.set_level(LogLevel.CRITICAL)

def get_default_base_dir():
    """Mendapatkan direktori default berdasarkan environment"""
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def on_save_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol save.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary berisi komponen UI
    """
    try:
        # Update status panel
        update_status_panel(ui_components, "Menyimpan konfigurasi backbone...", 'info')
        
        # Update konfigurasi dari UI
        config = update_config_from_ui(ui_components)
        
        # Simpan konfigurasi dan sinkronisasi
        saved_config = save_backbone_config(config, ui_components)
        
        # Update UI dari konfigurasi yang disimpan untuk memastikan konsistensi
        update_ui_from_config(ui_components, saved_config)
        
        # Update info panel dengan konfigurasi yang disimpan
        update_backbone_info(ui_components)
        
        # Tampilkan pesan sukses
        update_status_panel(ui_components, "Konfigurasi backbone berhasil disimpan", 'success')
        logger.info("Konfigurasi backbone berhasil disimpan dan UI diperbarui")
    except Exception as e:
        # Tampilkan pesan error
        update_status_panel(ui_components, f"Gagal menyimpan konfigurasi: {str(e)}", 'error')
        logger.error(f"Gagal menyimpan konfigurasi: {str(e)}")

def on_reset_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol reset.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary berisi komponen UI
    """
    try:
        # Update status panel
        update_status_panel(ui_components, "Mereset konfigurasi backbone...", 'info')
        
        # Reset konfigurasi ke default
        config = get_default_backbone_config()
        
        # Update UI dari konfigurasi default terlebih dahulu
        update_ui_from_config(ui_components, config)
        
        # Simpan konfigurasi default dan sinkronisasi
        saved_config = save_backbone_config(config, ui_components)
        
        # Update info panel
        update_backbone_info(ui_components)
        
        # Tampilkan pesan sukses
        update_status_panel(ui_components, "Konfigurasi backbone berhasil direset ke default", 'success')
        logger.info("Konfigurasi backbone berhasil direset ke default")
    except Exception as e:
        logger.error(f"Error saat reset konfigurasi backbone: {str(e)}")
        update_status_panel(ui_components, f"Error saat reset konfigurasi backbone: {str(e)}", 'error')