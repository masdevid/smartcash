"""
File: smartcash/ui/training_config/backbone/handlers/default_config.py
Deskripsi: Modul untuk mendefinisikan konfigurasi default backbone model
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def get_default_backbone_config() -> Dict[str, Any]:
    """
    Dapatkan konfigurasi default untuk backbone.
    
    Returns:
        Dictionary konfigurasi default
    """
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
            'normalization': {
                'type': 'batch_norm',
                'momentum': 0.1
            },
            'weights': {
                'path': '',
                'strict': True
            }
        }
    }

def get_backbone_options() -> Dict[str, Any]:
    """
    Dapatkan opsi-opsi untuk konfigurasi backbone.
    
    Returns:
        Dictionary berisi opsi-opsi konfigurasi
    """
    return {
        'backbones': [
            ('EfficientNet-B4', 'efficientnet_b4'),
            ('CSPDarknet-S', 'cspdarknet_s')
        ],
        'model_types': [
            ('EfficientNet Basic', 'efficient_basic'),
            ('YOLOv5s', 'yolov5s')
        ],
        'activation_functions': [
            ('ReLU', 'relu'),
            ('LeakyReLU', 'leaky_relu'),
            ('Mish', 'mish'),
            ('SiLU', 'silu')
        ],
        'normalization_types': [
            ('Batch Normalization', 'batch_norm'),
            ('Instance Normalization', 'instance_norm'),
            ('Group Normalization', 'group_norm')
        ]
    }