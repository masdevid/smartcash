"""
File: smartcash/ui/backbone/handlers/config_handler.py
Deskripsi: Fixed config handler dengan status panel update yang benar saat save/reset
"""

from typing import Dict, Any
from datetime import datetime
from smartcash.ui.handlers.config_handlers import ConfigHandler

class BackboneConfigHandler(ConfigHandler):
    """Config handler khusus untuk backbone configuration dengan fixed status updates"""
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract backbone configuration dari UI sesuai struktur backbone_config.yaml"""
        # One-liner safe getter dengan fallback
        get_val = lambda key, default: getattr(ui_components.get(key), 'value', default) if key in ui_components else default
        
        # Extract values sesuai mapping backbone_config.yaml
        backbone_val = get_val('backbone_dropdown', 'efficientnet_b4')
        model_type_val = get_val('model_type_dropdown', 'efficient_optimized')
        
        return {
            '_base_': 'base_config.yaml',
            
            # Konfigurasi backbone sesuai backbone_config.yaml
            'backbones': {
                backbone_val: {
                    'description': f'Selected {backbone_val} backbone',
                    'pretrained': True,
                    'features': 1792 if backbone_val == 'efficientnet_b4' else 1024,
                    'stages': [32, 56, 160, 1792] if backbone_val == 'efficientnet_b4' else [64, 128, 256, 1024]
                }
            },
            
            # Konfigurasi model_types sesuai backbone_config.yaml
            'model_types': {
                model_type_val: {
                    'description': f'Selected {model_type_val} model type',
                    'backbone': backbone_val,
                    'use_attention': get_val('use_attention_checkbox', False),
                    'use_residual': get_val('use_residual_checkbox', False),
                    'use_ciou': get_val('use_ciou_checkbox', False),
                    'detection_layers': ['banknote'],
                    'num_classes': 7,
                    'img_size': 640,
                    'pretrained': True
                }
            },
            
            # Feature adapter configuration
            'feature_adapter': {
                'channel_attention': get_val('use_attention_checkbox', False),
                'reduction_ratio': 16,
                'use_residual': get_val('use_residual_checkbox', False)
            },
            
            # Metadata
            'selected_backbone': backbone_val,
            'selected_model_type': model_type_val,
            'updated_at': datetime.now().isoformat(),
            'config_version': '1.0'
        }
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari backbone configuration dengan force reset checkbox state"""
        # Extract configuration values
        backbones = config.get('backbones', {})
        model_types = config.get('model_types', {})
        feature_adapter = config.get('feature_adapter', {})
        
        # Dapatkan selected values
        selected_backbone = config.get('selected_backbone', 'efficientnet_b4')
        selected_model_type = config.get('selected_model_type', 'efficient_optimized')
        
        # Get mapping untuk reset checkbox state yang benar
        from .defaults import get_model_type_mapping
        model_type_mapping = get_model_type_mapping()
        mapping = model_type_mapping.get(selected_model_type, {})
        
        # One-liner safe update function
        safe_update = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
        
        # Update dropdown first
        [safe_update(key, value) for key, value in [
            ('backbone_dropdown', selected_backbone),
            ('model_type_dropdown', selected_model_type)
        ]]
        
        # Force reset checkbox state berdasarkan model_type mapping
        checkbox_updates = [
            ('use_attention_checkbox', mapping.get('use_attention', False), mapping.get('disable_features', False)),
            ('use_residual_checkbox', mapping.get('use_residual', False), mapping.get('disable_features', False)),
            ('use_ciou_checkbox', mapping.get('use_ciou', False), mapping.get('disable_features', False))
        ]
        
        # Reset checkbox values dan disabled state
        for widget_name, value, disabled in checkbox_updates:
            if widget_name in ui_components:
                widget = ui_components[widget_name]
                widget.value = value  # Reset value sesuai mapping
                widget.disabled = disabled  # Set disabled untuk YOLOv5s
    
    def after_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Override dengan fixed status panel update"""
        backbone = config.get('selected_backbone', 'backbone')
        model_type = config.get('selected_model_type', 'model')
        self._update_status_panel(ui_components, f"Konfigurasi {backbone} + {model_type} tersimpan", "success")
        self.logger.success(f"💾 Backbone config tersimpan: {backbone} + {model_type}")
    
    def after_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Override dengan fixed status panel update"""
        backbone = config.get('selected_backbone', 'efficientnet_b4')
        self._update_status_panel(ui_components, f"Reset ke default: {backbone}", "success")
        self.logger.success(f"🔄 Backbone config direset ke default")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default backbone configuration"""
        from .defaults import get_default_backbone_config
        return get_default_backbone_config()