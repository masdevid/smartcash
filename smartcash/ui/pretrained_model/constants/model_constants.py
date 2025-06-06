"""
File: smartcash/ui/pretrained_model/constants/model_constants.py
Deskripsi: Constants untuk model URLs dan configurations dengan config integration
"""

from smartcash.common.config.manager import get_config_manager

# Default fallback values
DEFAULT_MODELS_DIR = '/content/models'
DEFAULT_DRIVE_MODELS_DIR = '/content/drive/MyDrive/SmartCash/models'

def get_model_configs() -> dict:
    """Get model configs dari pretrained_config.yaml dengan fallback"""
    try:
        config_manager = get_config_manager()
        config = config_manager.get_config('pretrained_config') or {}
        return config.get('pretrained_models', {}).get('models', _get_fallback_model_configs())
    except Exception:
        return _get_fallback_model_configs()

def _get_fallback_model_configs() -> dict:
    """Fallback model configurations"""
    return {
        'yolov5': {'name': 'YOLOv5s', 'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt',
                  'filename': 'yolov5s.pt', 'min_size_mb': 10, 'description': 'Object detection backbone'},
        'efficientnet_b4': {'name': 'EfficientNet-B4', 'url': 'https://huggingface.co/timm/efficientnet_b4.ra2_in1k/resolve/main/pytorch_model.bin',
                           'filename': 'efficientnet_b4_huggingface.bin', 'min_size_mb': 60, 'description': 'Feature extraction backbone'}
    }

# Dynamic property untuk CONFIG access
MODEL_CONFIGS = property(lambda self: get_model_configs())