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
        """🔍 Extract configuration from UI components
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            Dictionary containing the extracted configuration
        """
        from smartcash.common.logger import get_logger
        logger = get_logger(__name__)
        
        defaults = get_default_pretrained_config()['pretrained_models']
        config = {}
        
        for config_key, widget_key in self.config_mapping.items():
            try:
                widget = ui_components.get(widget_key)
                if widget is None:
                    config[config_key] = str(defaults.get(config_key, ''))
                    continue
                
                # Get value based on widget type
                value = getattr(widget, 'value', None)
                
                # Convert list to string if needed
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    value = str(value[0]) if value[0] is not None else ''
                else:
                    value = str(value) if value is not None else ''
                    
                config[config_key] = value
                
            except Exception as e:
                logger.error(f"Error processing {config_key} ({widget_key})", exc_info=True)
                config[config_key] = str(defaults.get(config_key, ''))
        
        return {'pretrained_models': config}
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """📝 Update UI components with config values
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration dictionary
        """
        from smartcash.common.logger import get_logger
        logger = get_logger(__name__)
        
        pretrained_config = config.get('pretrained_models', {})
        
        for config_key, widget_key in self.config_mapping.items():
            try:
                if widget_key not in ui_components:
                    continue
                    
                widget = ui_components[widget_key]
                value = pretrained_config.get(config_key)
                
                if value is None:
                    continue
                
                # Handle list values
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    value = value[0]  # Take first item if it's a list
                
                # Ensure we don't set None values
                if value is not None:
                    widget.value = value
                    
            except Exception as e:
                logger.error(f"Error updating {widget_key} ({config_key}): {str(e)}", exc_info=True)
    
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