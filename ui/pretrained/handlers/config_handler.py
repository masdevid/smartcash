"""
File: smartcash/ui/pretrained/handlers/config_handler.py
Deskripsi: Config handler untuk pretrained module dengan inheritance pattern
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import BaseConfigHandler

class PretrainedConfigHandler(BaseConfigHandler):
    """Config handler untuk pretrained dengan standardized pattern"""
    
    def __init__(self, config_name: str = 'pretrained_config'):
        super().__init__(config_name)
        self.logger.info("🔧 PretrainedConfigHandler initialized")
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components dengan optimized mapping"""
        try:
            # Get base config dari defaults
            config = self.get_default_config()
            
            # Extract dan update URLs
            url_mappings = {
                'yolov5_url_input': ('yolov5s', 'url'),
                'efficientnet_url_input': ('efficientnet_b4', 'url')
            }
            
            for widget_key, (model_key, field) in url_mappings.items():
                url_value = self._get_widget_value(ui_components, widget_key, '')
                if url_value:
                    config['pretrained_models']['models'][model_key][field] = url_value
            
            # Extract directories jika ada custom inputs
            dir_mappings = {
                'models_dir_input': 'models_dir',
                'drive_models_dir_input': 'drive_models_dir'
            }
            
            for widget_key, config_key in dir_mappings.items():
                dir_value = self._get_widget_value(ui_components, widget_key)
                if dir_value:
                    config['pretrained_models'][config_key] = dir_value
            
            self.logger.info("📝 Config extracted successfully")
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting config: {str(e)}")
            return self.get_default_config()
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """Update UI dengan config values"""
        try:
            pretrained_config = config.get('pretrained_models', {})
            models_config = pretrained_config.get('models', {})
            
            # Update YOLOv5 URL
            yolov5_config = models_config.get('yolov5s', {})
            self._set_widget_value(ui_components, 'yolov5_url_input', 
                                 yolov5_config.get('url', ''))
            
            # Update EfficientNet URL
            efficientnet_config = models_config.get('efficientnet_b4', {})
            self._set_widget_value(ui_components, 'efficientnet_url_input',
                                 efficientnet_config.get('url', ''))
            
            # Update directories jika ada
            self._set_widget_value(ui_components, 'models_dir_input',
                                 pretrained_config.get('models_dir', '/data/pretrained'))
            self._set_widget_value(ui_components, 'drive_models_dir_input',
                                 pretrained_config.get('drive_models_dir', 
                                                     '/content/drive/MyDrive/SmartCash/pretrained'))
            
            self.logger.info("🔄 UI updated dengan config values")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating UI: {str(e)}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Default config mapping dari defaults.py untuk consistency"""
        from smartcash.ui.pretrained.handlers.defaults import get_default_pretrained_config
        return get_default_pretrained_config()