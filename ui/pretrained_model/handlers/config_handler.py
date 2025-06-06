"""
File: smartcash/ui/pretrained_model/handlers/config_handler.py
Deskripsi: Handler konfigurasi untuk modul pretrained model dengan form integration
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler

class PretrainedModelConfigHandler(ConfigHandler):
    """Config handler untuk modul pretrained model dengan UI form extraction"""
    
    def __init__(self, module_name: str, parent_module: str = None):
        super().__init__(module_name, parent_module)
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI form components"""
        return {
            'pretrained_models': {
                'models_dir': getattr(ui_components.get('models_dir_input'), 'value', '/content/models'),
                'drive_models_dir': getattr(ui_components.get('drive_models_dir_input'), 'value', '/content/drive/MyDrive/SmartCash/models'),
                'models': {
                    'yolov5': {
                        'name': 'YOLOv5s',
                        'url': getattr(ui_components.get('yolov5_url_input'), 'value', ''),
                        'filename': 'yolov5s.pt',
                        'min_size_mb': 10,
                        'description': 'Object detection backbone'
                    },
                    'efficientnet_b4': {
                        'name': 'EfficientNet-B4',
                        'url': getattr(ui_components.get('efficientnet_url_input'), 'value', ''),
                        'filename': 'efficientnet_b4_huggingface.bin',
                        'min_size_mb': 60,
                        'description': 'Feature extraction backbone'
                    }
                }
            }
        }
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components dari config"""
        pretrained_config = config.get('pretrained_models', {})
        models_config = pretrained_config.get('models', {})
        
        # Update directory inputs dengan safe attribute setting
        models_dir_widget = ui_components.get('models_dir_input')
        models_dir_widget and setattr(models_dir_widget, 'value', pretrained_config.get('models_dir', '/content/models'))
        
        drive_dir_widget = ui_components.get('drive_models_dir_input')
        drive_dir_widget and setattr(drive_dir_widget, 'value', pretrained_config.get('drive_models_dir', '/content/drive/MyDrive/SmartCash/models'))
        
        # Update URL inputs dengan safe attribute setting
        yolov5_widget = ui_components.get('yolov5_url_input')
        yolov5_widget and setattr(yolov5_widget, 'value', models_config.get('yolov5', {}).get('url', ''))
        
        efficientnet_widget = ui_components.get('efficientnet_url_input')
        efficientnet_widget and setattr(efficientnet_widget, 'value', models_config.get('efficientnet_b4', {}).get('url', ''))
    
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