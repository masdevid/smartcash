# File: smartcash/ui/pretrained/handlers/config_handler.py
"""
File: smartcash/ui/pretrained/handlers/config_handler.py
Deskripsi: Safe config handler untuk pretrained models dengan robust validation
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.ui.handlers.config_handlers import BaseConfigHandler

logger = get_logger(__name__)

class PretrainedConfigHandler(BaseConfigHandler):
    """🔧 Safe config handler untuk pretrained models dengan validation"""
    
    def __init__(self, module_name: str = 'pretrained_models', parent_module: str = None):
        super().__init__(module_name, parent_module)
        
        # Safe mapping dari config key ke UI widget key
        self.config_mapping = {
            'models_dir': 'models_dir_text',
            'drive_models_dir': 'drive_models_dir_text', 
            'pretrained_type': 'pretrained_type_dropdown',
            'auto_download': 'auto_download_checkbox',
            'sync_drive': 'sync_drive_checkbox'
        }
    
    def safe_set_widget_value(self, widget, value, widget_name: str = "unknown") -> bool:
        """🛡️ Safely set widget value dengan comprehensive validation
        
        Args:
            widget: Widget object to update
            value: Value to set
            widget_name: Name untuk logging
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            # Check widget exists dan memiliki value attribute
            if not hasattr(widget, 'value'):
                logger.debug(f"Widget {widget_name} tidak memiliki value attribute")
                return False
            
            # Handle None values
            if value is None:
                logger.debug(f"Skipping None value untuk {widget_name}")
                return False
            
            # Type-specific validation dan conversion
            if hasattr(widget, 'options'):  # Dropdown widget
                return self._safe_set_dropdown_value(widget, value, widget_name)
            elif hasattr(widget, 'description') and 'checkbox' in widget_name.lower():  # Checkbox
                return self._safe_set_checkbox_value(widget, value, widget_name)
            else:  # Text widget
                return self._safe_set_text_value(widget, value, widget_name)
                
        except Exception as e:
            logger.warning(f"❌ Error setting {widget_name}: {str(e)}")
            return False
    
    def _safe_set_dropdown_value(self, widget, value, widget_name: str) -> bool:
        """Set dropdown value dengan validation"""
        try:
            # Convert lists to single value
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
                if value is None:
                    return False
            
            # Convert to string
            value_str = str(value).strip()
            
            # Check if value ada dalam options
            valid_values = [option[1] if isinstance(option, tuple) else option for option in widget.options]
            
            if value_str in valid_values:
                widget.value = value_str
                logger.debug(f"✅ Set {widget_name} = {value_str}")
                return True
            else:
                logger.debug(f"⚠️ Invalid value '{value_str}' untuk {widget_name}, valid: {valid_values}")
                return False
                
        except Exception as e:
            logger.warning(f"❌ Dropdown error {widget_name}: {str(e)}")
            return False
    
    def _safe_set_checkbox_value(self, widget, value, widget_name: str) -> bool:
        """Set checkbox value dengan safe boolean conversion"""
        try:
            # Convert various types to boolean
            if isinstance(value, bool):
                bool_value = value
            elif isinstance(value, str):
                bool_value = value.lower() in ('true', '1', 't', 'y', 'yes', 'on')
            elif isinstance(value, (int, float)):
                bool_value = bool(value)
            elif isinstance(value, (list, tuple)):
                bool_value = bool(value) and bool(value[0]) if value else False
            else:
                logger.debug(f"⚠️ Unknown type untuk checkbox {widget_name}: {type(value)}")
                return False
            
            widget.value = bool_value
            logger.debug(f"✅ Set {widget_name} = {bool_value}")
            return True
            
        except Exception as e:
            logger.warning(f"❌ Checkbox error {widget_name}: {str(e)}")
            return False
    
    def _safe_set_text_value(self, widget, value, widget_name: str) -> bool:
        """Set text value dengan safe string conversion"""
        try:
            # Convert to string safely
            if isinstance(value, (list, tuple)):
                str_value = str(value[0]) if value else ""
            else:
                str_value = str(value).strip()
            
            # Validate not empty
            if not str_value:
                logger.debug(f"⚠️ Empty value untuk {widget_name}")
                return False
            
            widget.value = str_value
            logger.debug(f"✅ Set {widget_name} = {str_value}")
            return True
            
        except Exception as e:
            logger.warning(f"❌ Text error {widget_name}: {str(e)}")
            return False
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """🔄 Update UI components dengan safe config values
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary to apply
        """
        try:
            if not isinstance(config, dict):
                logger.warning("❌ Config bukan dictionary, skip update")
                return
            
            pretrained_config = config.get('pretrained_models', {})
            if not isinstance(pretrained_config, dict):
                logger.warning("❌ Pretrained config bukan dictionary, skip update") 
                return
            
            success_count = 0
            total_count = 0
            
            # Update setiap mapping dengan safe validation
            for config_key, widget_key in self.config_mapping.items():
                total_count += 1
                
                # Check widget exists
                if widget_key not in ui_components:
                    logger.debug(f"⚠️ Widget {widget_key} tidak ditemukan")
                    continue
                
                # Get config value
                config_value = pretrained_config.get(config_key)
                if config_value is None:
                    logger.debug(f"⚠️ Config value untuk {config_key} adalah None")
                    continue
                
                # Safe set value
                widget = ui_components[widget_key]
                if self.safe_set_widget_value(widget, config_value, widget_key):
                    success_count += 1
            
            logger.info(f"✅ UI Update: {success_count}/{total_count} widgets berhasil diupdate")
            
        except Exception as e:
            logger.error(f"❌ Error updating UI: {str(e)}", exc_info=True)
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """📤 Extract config dari UI components dengan safe value extraction"""
        try:
            pretrained_config = {}
            
            # Extract values dengan safe getters
            for config_key, widget_key in self.config_mapping.items():
                if widget_key in ui_components:
                    widget = ui_components[widget_key]
                    if hasattr(widget, 'value'):
                        try:
                            pretrained_config[config_key] = widget.value
                        except Exception as e:
                            logger.warning(f"⚠️ Error extracting {widget_key}: {str(e)}")
            
            return {'pretrained_models': pretrained_config}
            
        except Exception as e:
            logger.error(f"❌ Error extracting config: {str(e)}")
            return {'pretrained_models': {}}
    
    def get_default_config(self) -> Dict[str, Any]:
        """📋 Get safe default configuration"""
        return {
            'pretrained_models': {
                'models_dir': '/content/models',
                'drive_models_dir': '/content/drive/MyDrive/SmartCash/models',
                'pretrained_type': 'yolov5s',
                'auto_download': False,
                'sync_drive': True
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """✅ Validate config dengan comprehensive checks"""
        try:
            pretrained_config = config.get('pretrained_models', {})
            errors = []
            
            # Validate required fields
            if not pretrained_config.get('models_dir'):
                errors.append("Models directory is required")
            
            # Validate model type
            valid_types = ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
            model_type = pretrained_config.get('pretrained_type')
            if model_type and model_type not in valid_types:
                errors.append(f"Invalid model type. Valid: {valid_types}")
            
            # Validate boolean fields
            for bool_field in ['auto_download', 'sync_drive']:
                value = pretrained_config.get(bool_field)
                if value is not None and not isinstance(value, bool):
                    try:
                        # Try convert to bool
                        bool(value)
                    except:
                        errors.append(f"Invalid boolean value for {bool_field}")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': []
            }
            
        except Exception as e:
            logger.error(f"❌ Validation error: {str(e)}")
            return {
                'valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': []
            }