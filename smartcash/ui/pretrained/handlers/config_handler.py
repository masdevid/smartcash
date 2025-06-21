"""
File: smartcash/ui/pretrained/handlers/config_handler.py
Deskripsi: Config handler untuk pretrained module dengan inheritance pattern yang diperbaiki
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import BaseConfigHandler

class PretrainedConfigHandler(BaseConfigHandler):
    """Config handler untuk pretrained dengan standardized pattern"""
    
    def __init__(self, module_name: str = 'pretrained', parent_module: str = None):
        """
        🔧 Inisialisasi PretrainedConfigHandler dengan parameter yang benar
        
        Args:
            module_name: Nama module (default: 'pretrained')  
            parent_module: Parent module jika ada (opsional)
        """
        # ✅ FIX: Gunakan parameter yang sesuai dengan BaseConfigHandler
        super().__init__(
            module_name=module_name,
            extract_fn=None,  # Akan menggunakan method extract_config dari class ini
            update_fn=None,   # Akan menggunakan method update_ui dari class ini
            parent_module=parent_module
        )
        self.logger.info(f"🔧 PretrainedConfigHandler initialized untuk module: {module_name}")
    
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
    
    def _get_widget_value(self, ui_components: Dict[str, Any], key: str, default=None):
        """Helper untuk extract widget value dengan safe handling"""
        try:
            widget = ui_components.get(key)
            if widget and hasattr(widget, 'value'):
                return widget.value
            return default
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting {key}: {str(e)}")
            return default
    
    def _set_widget_value(self, ui_components: Dict[str, Any], key: str, value):
        """Helper untuk set widget value dengan safe handling"""
        try:
            widget = ui_components.get(key)
            if widget and hasattr(widget, 'value'):
                widget.value = value
        except Exception as e:
            self.logger.warning(f"⚠️ Error setting {key}: {str(e)}")
    
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
        try:
            from smartcash.ui.pretrained.handlers.defaults import get_default_pretrained_config
            return get_default_pretrained_config()
        except Exception as e:
            self.logger.error(f"❌ Error loading defaults: {str(e)}")
            # Fallback config jika defaults.py tidak tersedia
            return {
                'pretrained_models': {
                    'models_dir': '/data/pretrained',
                    'drive_models_dir': '/content/drive/MyDrive/SmartCash/pretrained',
                    'models': {
                        'yolov5s': {
                            'name': 'YOLOv5s',
                            'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt',
                            'filename': 'yolov5s.pt',
                            'min_size_mb': 10,
                            'description': 'Object detection backbone'
                        },
                        'efficientnet_b4': {
                            'name': 'EfficientNet-B4',
                            'url': 'https://huggingface.co/timm/efficientnet_b4.ra2_in1k/resolve/main/pytorch_model.bin',
                            'filename': 'efficientnet_b4_huggingface.bin',
                            'min_size_mb': 60,
                            'description': 'Feature extraction backbone'
                        }
                    }
                }
            }