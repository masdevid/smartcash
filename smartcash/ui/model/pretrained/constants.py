"""
File: smartcash/ui/model/pretrained/constants.py
Constants for pretrained models module following UI module structure standard.
"""

from enum import Enum
from typing import Dict

# ==================== Enums ====================

class PretrainedModelType(Enum):
    """Available pretrained model types for download."""
    YOLOV5S = "yolov5s"
    EFFICIENTNET_B4 = "efficientnet_b4"

class PretrainedOperation(Enum):
    """Operations available in pretrained module."""
    DOWNLOAD = "download"

# ==================== Default Configurations ====================

DEFAULT_MODELS_DIR = "/data/pretrained"

DEFAULT_MODEL_URLS = {
    PretrainedModelType.YOLOV5S.value: "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
    PretrainedModelType.EFFICIENTNET_B4.value: ""  # Will use timm by default
}

EXPECTED_FILE_SIZES = {
    PretrainedModelType.YOLOV5S.value: 14_400_000,  # ~14.4MB
    PretrainedModelType.EFFICIENTNET_B4.value: 75_000_000  # ~75MB estimate
}

# ==================== Progress Steps ====================

PROGRESS_STEPS = {
    PretrainedOperation.DOWNLOAD.value: [
        "🔍 Checking existing models",
        "📁 Preparing download directory", 
        "📥 Downloading YOLOv5s model",
        "📥 Downloading EfficientNet-B4 model",
        "🔍 Validating downloaded models",
        "✅ Download complete"
    ]
}

# ==================== Module Information ====================

MODULE_GROUP: str = "model"
MODULE_NAME: str = "pretrained"
MODULE_TITLE: str = "Pretrained Models"
MODULE_DESCRIPTION: str = "Download and manage YOLOv5s and EfficientNet-B4 models"
MODULE_ICON: str = "🤖"
MODULE_VERSION: str = "2.0.0"

# ==================== UI Configuration ====================

UI_CONFIG = {
    'title': MODULE_TITLE,  # Required by UI components
    'subtitle': MODULE_DESCRIPTION,  # Required by UI components
    'module_title': MODULE_TITLE,
    'module_description': MODULE_DESCRIPTION,
    'module_icon': MODULE_ICON,
    'module_name': MODULE_NAME,
    'parent_module': MODULE_GROUP,
    'version': MODULE_VERSION
}

# Module Metadata
MODULE_METADATA = {
    'name': 'pretrained',
    'title': 'Pretrained Models',
    'description': 'Pretrained model management module with one-click setup and refresh operations',
    'version': '2.0.0',
    'category': 'model',
    'author': 'SmartCash',
    'tags': ['pretrained', 'models', 'yolov5', 'efficientnet', 'download'],
    'features': [
        'YOLOv5s model download',
        'EfficientNet-B4 model download',
        'Model existence validation',
        'Download progress tracking',
        'Cleanup corrupted models',
        'Model integrity verification'
    ]
}

# Button Configuration
BUTTON_CONFIG = {
    'download': {
        'text': '📥 Download Models',
        'style': 'success',
        'tooltip': 'Download YOLOv5s and EfficientNet-B4 pretrained models',
        'order': 1
    },
    'validate': {
        'text': '🔍 Validate Models',
        'style': 'info',
        'tooltip': 'Validate downloaded models and check integrity',
        'order': 2
    },
    'refresh': {
        'text': '🔄 Refresh Status',
        'style': 'warning',
        'tooltip': 'Refresh model status and directory contents',
        'order': 3
    },
    'cleanup': {
        'text': '🗑️ Clean Up',
        'style': 'danger',
        'tooltip': 'Remove corrupted or invalid model files',
        'order': 4
    }
}

# ==================== Model Information ====================

MODEL_INFO = {
    PretrainedModelType.YOLOV5S.value: {
        "name": "YOLOv5s",
        "description": "YOLOv5 Small - Fast and efficient object detection",
        "source": "Ultralytics GitHub",
        "file_extension": ".pt",
        "download_method": "direct"
    },
    PretrainedModelType.EFFICIENTNET_B4.value: {
        "name": "EfficientNet-B4",
        "description": "EfficientNet-B4 - Efficient CNN backbone",
        "source": "PyTorch Hub / timm",
        "file_extension": ".pth",
        "download_method": "timm"
    }
}

# ==================== Validation Settings ====================

VALIDATION_CONFIG = {
    "min_file_size": 1024,  # 1KB minimum
    "size_tolerance": 0.2,  # 20% tolerance for file size validation
    "required_extensions": [".pt", ".pth"],
    "timeout_seconds": 300  # 5 minutes timeout for downloads
}

# ==================== Defaults ====================

DEFAULT_CONFIG = {
    "models_dir": DEFAULT_MODELS_DIR,
    "model_urls": DEFAULT_MODEL_URLS.copy(),
    "auto_download": False,
    "validate_downloads": True,
    "cleanup_failed": True
}