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
        from smartcash.common.logger import get_logger
        logger = get_logger(__name__)
        
        defaults = get_default_pretrained_config()['pretrained_models']
        config = {}
        
        logger.debug(f"Extracting config from UI components. Available keys: {list(ui_components.keys())}")
        
        for config_key, widget_key in self.config_mapping.items():
            try:
                widget = ui_components.get(widget_key)
                if widget is None:
                    logger.warning(f"Widget {widget_key} not found in UI components")
                    value = defaults.get(config_key, '')
                else:
                    value = getattr(widget, 'value', defaults.get(config_key, ''))
                
                logger.debug(f"Processing {config_key} (widget: {widget_key}), value: {value}, type: {type(value)}")
                
                # Ensure value is a string, not a list
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    logger.debug(f"Converting list to string for {config_key}: {value}")
                    value = str(value[0]) if value[0] is not None else ''
                else:
                    value = str(value) if value is not None else ''
                    
                config[config_key] = value
                
            except Exception as e:
                logger.error(f"Error processing {config_key} ({widget_key}): {str(e)}", exc_info=True)
                config[config_key] = str(defaults.get(config_key, ''))
        
        logger.debug(f"Final config: {config}")
        return {'pretrained_models': config}
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """📝 Update dengan safe access"""
        from smartcash.common.logger import get_logger
        logger = get_logger(__name__)
        
        pretrained_config = config.get('pretrained_models', {})
        logger.debug(f"Updating UI with config: {pretrained_config}")
        
        for config_key, widget_key in self.config_mapping.items():
            try:
                if widget_key not in ui_components:
                    logger.warning(f"Widget {widget_key} not found in UI components")
                    continue
                    
                widget = ui_components[widget_key]
                if not hasattr(widget, 'value'):
                    logger.warning(f"Widget {widget_key} has no 'value' attribute")
                    continue
                    
                value = pretrained_config.get(config_key)
                logger.debug(f"Setting {widget_key} ({config_key}) to: {value}, type: {type(value)}")
                
                # Handle list values
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    value = value[0]  # Take first item if it's a list
                    logger.debug(f"Converted list to single value: {value}")
                
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