"""
File: smartcash/ui/training_config/backbone/handlers/button_handlers.py
Deskripsi: Handler untuk tombol UI pada komponen backbone
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, HTML

def setup_backbone_button_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol pada komponen UI backbone.
    
    Args:
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI dengan handler terpasang
    """
    try:
        # Import dengan penanganan error minimal
        from smartcash.ui.training_config.config_handler import save_config, reset_config
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        # Dapatkan logger jika tersedia
        logger = ui_components.get('logger', None)
        
        # Validasi config
        if config is None: config = {}
        
        # Default config
        default_config = {
            'model': {
                'backbone': 'efficientnet_b4',
                'pretrained': True,
                'freeze_backbone': True,
                'freeze_layers': 3
            }
        }
        
        # Update config dari UI
        def update_config_from_ui(current_config=None):
            if current_config is None: current_config = config
            
            # Update config dari nilai UI
            if 'model' not in current_config:
                current_config['model'] = {}
                
            current_config['model']['backbone'] = ui_components['backbone_type'].value
            current_config['model']['pretrained'] = ui_components['pretrained'].value
            current_config['model']['freeze_backbone'] = ui_components['freeze_backbone'].value
            current_config['model']['freeze_layers'] = ui_components['freeze_layers'].value
            
            # Update info backbone
            update_backbone_info()
            
            return current_config
        
        # Update UI dari config
        def update_ui_from_config():
            if not config or 'model' not in config: return
            
            try:
                # Update nilai UI dari config
                if 'backbone' in config['model']:
                    ui_components['backbone_type'].value = config['model']['backbone']
                    
                if 'pretrained' in config['model']:
                    ui_components['pretrained'].value = config['model']['pretrained']
                    
                if 'freeze_backbone' in config['model']:
                    ui_components['freeze_backbone'].value = config['model']['freeze_backbone']
                    
                if 'freeze_layers' in config['model']:
                    ui_components['freeze_layers'].value = config['model']['freeze_layers']
                
                # Update info backbone
                update_backbone_info()
                
                if logger: logger.info("✅ UI backbone diperbarui dari config")
            except Exception as e:
                if logger: logger.error(f"❌ Error update UI: {e}")
        
        # Update informasi backbone
        def update_backbone_info():
            try:
                from smartcash.model.config.backbone_config import BackboneConfig
                
                # Buat instance BackboneConfig
                backbone_type = ui_components['backbone_type'].value
                backbone_config = BackboneConfig(backbone_type=backbone_type)
                
                # Dapatkan informasi backbone
                info_html = f"""
                <h4>Informasi Backbone: {backbone_type}</h4>
                <ul>
                    <li><b>Stride:</b> {backbone_config.stride}</li>
                    <li><b>Features:</b> {backbone_config.features}</li>
                    <li><b>Stages:</b> {backbone_config.stages}</li>
                    <li><b>Width Coefficient:</b> {backbone_config.width_coefficient}</li>
                    <li><b>Depth Coefficient:</b> {backbone_config.depth_coefficient}</li>
                    <li><b>Pretrained:</b> {'Ya' if backbone_config.pretrained else 'Tidak'}</li>
                </ul>
                """
                
                ui_components['backbone_info'].value = info_html
            except Exception as e:
                ui_components['backbone_info'].value = f"<p style='color:red'>❌ Error: {str(e)}</p>"
        
        # Handler buttons
        def on_save_click(b): 
            save_config(ui_components, config, "configs/model_config.yaml", update_config_from_ui, "Model Backbone")
        
        def on_reset_click(b): 
            reset_config(ui_components, config, default_config, update_ui_from_config, "Model Backbone")
        
        # Register handlers
        ui_components['save_button'].on_click(on_save_click)
        ui_components['reset_button'].on_click(on_reset_click)
        
        # Tambahkan fungsi ke ui_components
        ui_components['update_config_from_ui'] = update_config_from_ui
        ui_components['update_ui_from_config'] = update_ui_from_config
        ui_components['update_backbone_info'] = update_backbone_info
        
        # Inisialisasi UI dari config
        update_ui_from_config()
        
    except Exception as e:
        # Fallback sederhana jika terjadi error
        if 'status' in ui_components:
            with ui_components['status']: display(HTML(f"<p style='color:red'>❌ Error setup backbone button handler: {str(e)}</p>"))
        else: print(f"❌ Error setup backbone button handler: {str(e)}")
    
    return ui_components
