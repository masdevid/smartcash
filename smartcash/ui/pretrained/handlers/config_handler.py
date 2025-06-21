# File: smartcash/ui/pretrained/handlers/config_handler.py
"""
File: smartcash/ui/pretrained/handlers/config_handler.py  
Deskripsi: Config handler untuk pretrained module dengan ConfigHandler inheritance - Fixed constructor
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.pretrained.handlers.defaults import get_default_pretrained_config

class PretrainedConfigHandler(ConfigHandler):
    """🔧 Config handler untuk pretrained module dengan ConfigHandler inheritance - Fixed constructor"""
    
    def __init__(self, module_name: str = 'pretrained_models', parent_module: str = None):
        """Fixed constructor untuk accept parameters dari CommonInitializer"""
        super().__init__(module_name, parent_module)
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
        config = {}
        
        for config_key, widget_key in self.config_mapping.items():
            value = getattr(ui_components.get(widget_key), 'value', defaults.get(config_key, ''))
            # Ensure pretrained_type is a string, not a list
            if config_key == 'pretrained_type' and isinstance(value, (list, tuple)) and len(value) > 0:
                value = value[0]
            config[config_key] = str(value) if value is not None else ''
            
        return {'pretrained_models': config}
    
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