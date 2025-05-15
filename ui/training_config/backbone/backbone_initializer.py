"""
File: smartcash/ui/training_config/backbone/backbone_initializer.py
Deskripsi: Inisialisasi UI dan logika bisnis untuk pemilihan backbone model SmartCash
"""

from typing import Dict, Any
from IPython.display import display, clear_output

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.common.config.manager import ConfigManager

logger = get_logger(__name__)

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
        
        # Setup handler untuk sinkronisasi dengan Drive
        from smartcash.ui.training_config.backbone.handlers.drive_handlers import (
            sync_to_drive,
            sync_from_drive
        )
        
        # Register handler untuk tombol
        ui_components['save_button'].on_click(
            lambda b: on_save_click(b, ui_components)
        )
        
        ui_components['reset_button'].on_click(
            lambda b: on_reset_click(b, ui_components)
        )
        
        # Register handler untuk tombol sinkronisasi dengan Drive
        ui_components['sync_to_drive_button'].on_click(
            lambda b: sync_to_drive(b, ui_components)
        )
        
        ui_components['sync_from_drive_button'].on_click(
            lambda b: sync_from_drive(b, ui_components)
        )
        
        # Inisialisasi UI dari konfigurasi
        from smartcash.ui.training_config.backbone.handlers.config_handlers import (
            update_ui_from_config,
            update_backbone_info
        )
        
        # Dapatkan ConfigManager
        config_manager = ConfigManager.get_instance()
        
        # Dapatkan konfigurasi
        current_config = config_manager.get_module_config('model')
        
        # Update UI dari konfigurasi
        update_ui_from_config(ui_components, current_config)
        
        # Update info panel
        update_backbone_info(ui_components)
        
        # Register UI components untuk persistensi
        config_manager.register_ui_components('backbone', ui_components)
        
        logger.info(f"{ICONS.get('success', '✅')} UI backbone berhasil diinisialisasi")
        
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
