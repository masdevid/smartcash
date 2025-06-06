"""
File: smartcash/ui/pretrained_model/pretrained_initializer.py
Deskripsi: Optimized initializer dengan UI form integration dan config handler
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import PRETRAINED_MODEL_LOGGER_NAMESPACE, KNOWN_NAMESPACES
from smartcash.ui.pretrained_model.components.ui_components import create_pretrained_main_ui
from smartcash.ui.pretrained_model.handlers.pretrained_handlers import setup_pretrained_handlers
from smartcash.ui.pretrained_model.handlers.config_handler import PretrainedModelConfigHandler

MODULE_LOGGER_NAME = KNOWN_NAMESPACES[PRETRAINED_MODEL_LOGGER_NAMESPACE]

class PretrainedModelInitializer(CommonInitializer):
    """Optimized initializer dengan UI form integration"""
    
    def __init__(self):
        super().__init__(MODULE_LOGGER_NAME, PretrainedModelConfigHandler)
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan form integration"""
        ui_components = create_pretrained_main_ui(config)
        ui_components.update({'pretrained_model_initialized': True})
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan UI integration"""
        return setup_pretrained_handlers(ui_components, config, env)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default config dari pretrained_config.yaml"""
        return {
            'pretrained_models': {
                'models_dir': '/content/models',
                'drive_models_dir': '/content/drive/MyDrive/SmartCash/models',
                'models': {
                    'yolov5': {'name': 'YOLOv5s', 'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt',
                              'filename': 'yolov5s.pt', 'min_size_mb': 10, 'description': 'Object detection backbone'},
                    'efficientnet_b4': {'name': 'EfficientNet-B4', 'url': 'https://huggingface.co/timm/efficientnet_b4.ra2_in1k/resolve/main/pytorch_model.bin',
                                       'filename': 'efficientnet_b4_huggingface.bin', 'min_size_mb': 60, 'description': 'Feature extraction backbone'}
                }
            }
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical components untuk validasi"""
        return ['ui', 'download_sync_button', 'log_output', 'status_panel', 'progress_tracker', 'config_form']

# Global instance dan public API
_pretrained_initializer = PretrainedModelInitializer()
initialize_pretrained_model_ui = lambda env=None, config=None: _pretrained_initializer.initialize(env=env, config=config)