"""
File: smartcash/ui/pretrained_model/handlers/config_handler.py
Deskripsi: Handler konfigurasi untuk modul pretrained model dengan form integration
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import BaseConfigHandler

class PretrainedModelConfigHandler(BaseConfigHandler):
    """Config handler untuk modul pretrained model dengan UI form extraction"""
    
    def __init__(self, module_name: str, parent_module: str = None):
        super().__init__(module_name, self._extract_config_from_ui, self._update_ui_from_config, parent_module)
    
    def _extract_config_from_ui(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI form components"""
        return {
            'pretrained_models': {
                'models_dir': ui_components.get('models_dir_input', {}).get('value', '/content/models'),
                'drive_models_dir': ui_components.get('drive_models_dir_input', {}).get('value', '/content/drive/MyDrive/SmartCash/models'),
                'models': {
                    'yolov5': {
                        'name': 'YOLOv5s',
                        'url': ui_components.get('yolov5_url_input', {}).get('value', ''),
                        'filename': 'yolov5s.pt',
                        'min_size_mb': 10,
                        'description': 'Object detection backbone'
                    },
                    'efficientnet_b4': {
                        'name': 'EfficientNet-B4',
                        'url': ui_components.get('efficientnet_url_input', {}).get('value', ''),
                        'filename': 'efficientnet_b4_huggingface.bin',
                        'min_size_mb': 60,
                        'description': 'Feature extraction backbone'
                    }
                }
            }
        }
    
    def _update_ui_from_config(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components dari config"""
        pretrained_config = config.get('pretrained_models', {})
        models_config = pretrained_config.get('models', {})
        
        # Update directory inputs
        ui_components.get('models_dir_input') and setattr(ui_components['models_dir_input'], 'value', 
                                                         pretrained_config.get('models_dir', '/content/models'))
        ui_components.get('drive_models_dir_input') and setattr(ui_components['drive_models_dir_input'], 'value', 
                                                               pretrained_config.get('drive_models_dir', '/content/drive/MyDrive/SmartCash/models'))
        
        # Update URL inputs
        ui_components.get('yolov5_url_input') and setattr(ui_components['yolov5_url_input'], 'value', 
                                                         models_config.get('yolov5', {}).get('url', ''))
        ui_components.get('efficientnet_url_input') and setattr(ui_components['efficientnet_url_input'], 'value', 
                                                               models_config.get('efficientnet_b4', {}).get('url', ''))
    
    def get_default_config(self) -> Dict[str, Any]:
        """Default config untuk pretrained model"""
        return {
            'pretrained_models': {
                'models_dir': '/content/models',
                'drive_models_dir': '/content/drive/MyDrive/SmartCash/models',
                'models': {
                    'yolov5': {
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