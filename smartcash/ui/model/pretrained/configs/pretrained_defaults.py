"""
File: smartcash/ui/model/pretrained/configs/pretrained_defaults.py
Default configuration values for pretrained models module.
"""

from typing import Dict, Any
from ..constants import DEFAULT_CONFIG, PretrainedModelType, DEFAULT_MODEL_URLS

def get_pretrained_defaults() -> Dict[str, Any]:
    """
    Get default configuration for pretrained models module.
    
    Returns:
        Dictionary containing default configuration values
    """
    return {
        "models_dir": "/data/pretrained",
        "model_urls": {
            PretrainedModelType.YOLOV5S.value: DEFAULT_MODEL_URLS[PretrainedModelType.YOLOV5S.value],
            PretrainedModelType.EFFICIENTNET_B4.value: DEFAULT_MODEL_URLS[PretrainedModelType.EFFICIENTNET_B4.value]
        },
        "auto_download": False,
        "validate_downloads": True,
        "cleanup_failed": True,
        "download_timeout": 300,  # 5 minutes
        "chunk_size": 8192,
        "progress_update_interval": 1024 * 1024  # Update progress every 1MB
    }

def get_yaml_schema() -> Dict[str, Any]:
    """
    Get YAML schema definition for pretrained models configuration.
    
    Returns:
        Dictionary containing YAML schema
    """
    return {
        "pretrained_models": {
            "description": "Configuration for pretrained models download and management",
            "type": "dict",
            "properties": {
                "models_dir": {
                    "type": "string",
                    "description": "Directory to store downloaded pretrained models",
                    "default": "/data/pretrained"
                },
                "model_urls": {
                    "type": "dict",
                    "description": "Custom URLs for model downloads",
                    "properties": {
                        "yolov5s": {
                            "type": "string",
                            "description": "Custom URL for YOLOv5s model download"
                        },
                        "efficientnet_b4": {
                            "type": "string", 
                            "description": "Custom URL for EfficientNet-B4 model download"
                        }
                    }
                },
                "auto_download": {
                    "type": "boolean",
                    "description": "Automatically download missing models",
                    "default": False
                },
                "validate_downloads": {
                    "type": "boolean",
                    "description": "Validate downloaded model files",
                    "default": True
                },
                "cleanup_failed": {
                    "type": "boolean",
                    "description": "Remove failed/incomplete downloads",
                    "default": True
                }
            }
        }
    }