"""
File: smartcash/ui/training_config/backbone/handlers/form_handlers.py
Deskripsi: Handler untuk form UI pada komponen backbone
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, HTML

def setup_backbone_form_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk form pada komponen UI backbone.
    
    Args:
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI dengan handler terpasang
    """
    try:
        # Dapatkan logger jika tersedia
        logger = ui_components.get('logger', None)
        
        # Handler untuk perubahan komponen
        def on_backbone_type_change(change):
            if change['name'] == 'value':
                # Update info backbone
                if 'update_backbone_info' in ui_components:
                    ui_components['update_backbone_info']()
                # Update config
                if 'update_config_from_ui' in ui_components:
                    ui_components['update_config_from_ui']()
        
        def on_pretrained_change(change):
            if change['name'] == 'value':
                # Update config
                if 'update_config_from_ui' in ui_components:
                    ui_components['update_config_from_ui']()
        
        def on_freeze_backbone_change(change):
            if change['name'] == 'value':
                # Aktifkan/nonaktifkan slider freeze_layers
                ui_components['freeze_layers'].disabled = not change['new']
                # Update config
                if 'update_config_from_ui' in ui_components:
                    ui_components['update_config_from_ui']()
        
        def on_freeze_layers_change(change):
            if change['name'] == 'value':
                # Update config
                if 'update_config_from_ui' in ui_components:
                    ui_components['update_config_from_ui']()
        
        # Register observers
        ui_components['backbone_type'].observe(on_backbone_type_change)
        ui_components['pretrained'].observe(on_pretrained_change)
        ui_components['freeze_backbone'].observe(on_freeze_backbone_change)
        ui_components['freeze_layers'].observe(on_freeze_layers_change)
        
        # Disable freeze_layers jika freeze_backbone tidak aktif
        ui_components['freeze_layers'].disabled = not ui_components['freeze_backbone'].value
        
        # Cleanup function
        def cleanup():
            try:
                # Hapus semua observer
                ui_components['backbone_type'].unobserve(on_backbone_type_change)
                ui_components['pretrained'].unobserve(on_pretrained_change)
                ui_components['freeze_backbone'].unobserve(on_freeze_backbone_change)
                ui_components['freeze_layers'].unobserve(on_freeze_layers_change)
                
                if logger: logger.info("✅ Backbone form handlers cleaned up")
            except Exception as e:
                if logger: logger.warning(f"⚠️ Error cleanup: {e}")
        
        # Tambahkan cleanup function
        ui_components['cleanup'] = cleanup
        
    except Exception as e:
        # Fallback sederhana jika terjadi error
        if 'status' in ui_components:
            with ui_components['status']: display(HTML(f"<p style='color:red'>❌ Error setup backbone form handler: {str(e)}</p>"))
        else: print(f"❌ Error setup backbone form handler: {str(e)}")
    
    return ui_components
