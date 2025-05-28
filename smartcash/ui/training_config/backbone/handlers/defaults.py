"""
File: smartcash/ui/training_config/backbone/handlers/defaults.py
Deskripsi: Default backbone configuration values dengan one-liner structure
"""

from typing import Dict, Any

def get_default_backbone_config() -> Dict[str, Any]:
    """Get default backbone configuration dengan complete structure"""
    return {
        'model': {
            'enabled': True,
            'backbone': 'efficientnet_b4',
            'model_type': 'efficient_basic', 
            'use_attention': True,
            'use_residual': True,
            'use_ciou': False,
            'pretrained': True,
            'freeze_backbone': False,
            'freeze_bn': False,
            'dropout': 0.2,
            'activation': 'relu',
            'normalization': {'type': 'batch_norm', 'momentum': 0.1},
            'weights': {'path': '', 'strict': True}
        },
        'config_version': '1.0',
        'description': 'Default backbone configuration untuk SmartCash detection'
    }

# One-liner option getters
get_backbone_options = lambda: [('EfficientNet-B4', 'efficientnet_b4'), ('CSPDarknet-S', 'cspdarknet_s')]
get_model_type_options = lambda: [('EfficientNet Basic', 'efficient_basic'), ('YOLOv5s', 'yolov5s')]
get_optimization_features = lambda: [('FeatureAdapter (Attention)', 'use_attention'), ('ResidualAdapter (Residual)', 'use_residual'), ('CIoU Loss', 'use_ciou')]