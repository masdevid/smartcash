"""
File: smartcash/ui/training_config/backbone/backbone_initializer.py
Deskripsi: Inisialisasi UI dan logika bisnis untuk pemilihan backbone model SmartCash
"""

from typing import Dict, Any
from IPython.display import display, clear_output
from pathlib import Path
import os

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.common.config import get_config_manager
from smartcash.common.environment import get_environment_manager

logger = get_logger(__name__)

def get_default_base_dir():
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def initialize_backbone_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI untuk pemilihan backbone model.
    
    Returns:
        Dictionary berisi komponen UI yang telah diinisialisasi
    """
    try:
        # Import komponen UI
        from smartcash.ui.training_config.backbone.components.backbone_components import create_backbone_ui
        
        # Buat komponen UI
        ui_components = create_backbone_ui()
        
        # Tambahkan status panel
        from smartcash.ui.training_config.backbone.handlers.status_handlers import add_status_panel
        ui_components = add_status_panel(ui_components)
        
        # Tampilkan UI
        clear_output(wait=True)
        display(ui_components['main_container'])
        
        # Setup handler untuk form
        from smartcash.ui.training_config.backbone.handlers.form_handlers import (
            on_backbone_change,
            on_model_type_change,
            on_attention_change,
            on_residual_change,
            on_ciou_change
        )
        
        # Register handler untuk form
        ui_components['backbone_dropdown'].observe(
            lambda change: on_backbone_change(change, ui_components),
            names='value'
        )
        
        ui_components['model_type_dropdown'].observe(
            lambda change: on_model_type_change(change, ui_components),
            names='value'
        )
        
        ui_components['use_attention_checkbox'].observe(
            lambda change: on_attention_change(change, ui_components),
            names='value'
        )
        
        ui_components['use_residual_checkbox'].observe(
            lambda change: on_residual_change(change, ui_components),
            names='value'
        )
        
        ui_components['use_ciou_checkbox'].observe(
            lambda change: on_ciou_change(change, ui_components),
            names='value'
        )
        
        # Setup handler untuk tombol
        from smartcash.ui.training_config.backbone.handlers.button_handlers import (
            on_save_click,
            on_reset_click
        )
        
        # Environment manager sudah diimpor di bagian atas file
        
        # Register handler untuk tombol
        ui_components['save_button'].on_click(
            lambda b: on_save_click(b, ui_components)
        )
        
        ui_components['reset_button'].on_click(
            lambda b: on_reset_click(b, ui_components)
        )
        
        # Perbarui informasi sinkronisasi berdasarkan status drive
        try:
            env_manager = get_environment_manager(base_dir=get_default_base_dir())
            if not env_manager.is_drive_mounted:
                # Jika drive tidak diaktifkan, perbarui pesan
                ui_components['sync_info'].value = f"<div style='margin-top: 5px; font-style: italic; color: #666;'>{ICONS.get('warning', '⚠️')} Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi otomatis.</div>"
        except Exception as e:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal memeriksa status drive: {str(e)}")
        
        # Inisialisasi UI dari konfigurasi
        from smartcash.ui.training_config.backbone.handlers.config_handlers import (
            update_ui_from_config,
            update_backbone_info
        )
        
        # Dapatkan ConfigManager
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        
        # Dapatkan konfigurasi
        current_config = config_manager.get_module_config('model')
        
        # Update UI dari konfigurasi
        update_ui_from_config(ui_components, current_config)
        
        # Update info panel
        update_backbone_info(ui_components)
        
        # Register UI components untuk persistensi
        config_manager.register_ui_components('backbone', ui_components)
        
        # Update status panel dengan pesan selamat datang
        from smartcash.ui.training_config.backbone.handlers.status_handlers import update_status_panel
        update_status_panel(ui_components, "Konfigurasi backbone siap digunakan", 'info')
        
        return ui_components
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat inisialisasi UI backbone: {str(e)}")
        
        # Import widgets jika belum diimpor
        import ipywidgets as widgets
        
        # Buat container minimal untuk menampilkan error
        error_container = widgets.VBox([
            widgets.HTML(f"<h3>{ICONS.get('error', '❌')} Error saat inisialisasi UI backbone</h3>"),
            widgets.HTML(f"<p>{str(e)}</p>")
        ])
        
        display(error_container)
        
        return {'main_container': error_container}
