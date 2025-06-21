# File: smartcash/ui/pretrained/handlers/config_handler.py
"""
File: smartcash/ui/pretrained/handlers/config_handler.py  
Deskripsi: Config handler untuk pretrained module dengan ConfigHandler inheritance
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.pretrained.handlers.defaults import get_default_pretrained_config

class PretrainedConfigHandler(ConfigHandler):
    """🔧 Config handler untuk pretrained module dengan ConfigHandler inheritance"""
    
    def __init__(self):
        super().__init__('pretrained_models', None)
        self.config_mapping = {
            'models_dir': 'models_dir_input',
            'drive_models_dir': 'drive_models_dir_input', 
            'pretrained_type': 'pretrained_type_dropdown',
            'auto_download': 'auto_download_checkbox',
            'sync_drive': 'sync_drive_checkbox'
        }
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """🔍 Extract dengan fallback ke defaults"""
        defaults = get_default_pretrained_config()['pretrained_models']
        return {'pretrained_models': {
            config_key: getattr(ui_components.get(widget_key), 'value', defaults[config_key])
            for config_key, widget_key in self.config_mapping.items()
        }}
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """📝 Update dengan safe access"""
        pretrained_config = config.get('pretrained_models', {})
        [setattr(ui_components[widget_key], 'value', pretrained_config.get(config_key))
         for config_key, widget_key in self.config_mapping.items()
         if widget_key in ui_components and hasattr(ui_components[widget_key], 'value')]
    
    def get_default_config(self) -> Dict[str, Any]:
        """📋 Delegate ke defaults.py"""
        return get_default_pretrained_config()
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """✅ Streamlined validation"""
        pretrained_config = config.get('pretrained_models', {})
        valid_types = ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
        
        errors = [error for error in [
            "Models directory required" if not pretrained_config.get('models_dir') else None,
            f"Invalid type. Use: {valid_types}" if pretrained_config.get('pretrained_type') not in valid_types else None
        ] if error]
        
        return {'valid': not errors, 'errors': errors}