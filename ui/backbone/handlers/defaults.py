"""
File: smartcash/ui/backbone/handlers/defaults.py
Deskripsi: Default configuration untuk backbone berdasarkan backbone_config.yaml
"""

from typing import Dict, Any, List, Tuple

def get_default_backbone_config() -> Dict[str, Any]:
    """Default backbone configuration sesuai backbone_config.yaml structure"""
    return {
        '_base_': 'base_config.yaml',
        
        'backbones': {
            'efficientnet_b4': {
                'description': 'EfficientNet-B4 backbone dengan guaranteed 3 feature maps output untuk FPN-PAN',
                'stride': 32,
                'width_coefficient': 1.4,
                'depth_coefficient': 1.8,
                'features': 1792,
                'stages': [32, 56, 160, 1792],
                'feature_indices': [2, 3, 4],
                'out_channels': [128, 256, 512],
                'pretrained': True
            },
            'cspdarknet_s': {
                'description': 'CSPDarknet-Small backbone yang digunakan oleh YOLOv5s',
                'stride': 32,
                'width_coefficient': 1.0,
                'depth_coefficient': 1.0,
                'features': 1024,
                'stages': [64, 128, 256, 1024],
                'pretrained': True
            }
        },
        
        'model_types': {
            'yolov5s': {
                'description': 'YOLOv5s dengan CSPDarknet sebagai backbone (model pembanding)',
                'backbone': 'cspdarknet_s',
                'use_attention': False,
                'use_residual': False,
                'use_ciou': False,
                'detection_layers': ['banknote'],
                'num_classes': 7,
                'img_size': 640,
                'pretrained': True
            },
            'efficient_basic': {
                'description': 'Model dasar tanpa optimasi khusus',
                'backbone': 'efficientnet_b4',
                'use_attention': False,
                'use_residual': False,
                'use_ciou': False,
                'detection_layers': ['banknote'],
                'num_classes': 7,
                'img_size': 640,
                'pretrained': True
            },
            'efficient_optimized': {
                'description': 'Model dengan EfficientNet-B4 dan FeatureAdapter',
                'backbone': 'efficientnet_b4',
                'use_attention': True,
                'use_residual': False,
                'use_ciou': False,
                'detection_layers': ['banknote'],
                'num_classes': 7,
                'img_size': 640,
                'pretrained': True
            },
            'efficient_advanced': {
                'description': 'Model dengan semua optimasi: FeatureAdapter, ResidualAdapter, dan CIoU',
                'backbone': 'efficientnet_b4',
                'use_attention': True,
                'use_residual': True,
                'use_ciou': True,
                'detection_layers': ['banknote'],
                'num_classes': 7,
                'img_size': 640,
                'pretrained': True
            }
        },
        
        'feature_adapter': {
            'channel_attention': True,
            'reduction_ratio': 16,
            'use_residual': False
        },
        
        'selected_backbone': 'efficientnet_b4',
        'selected_model_type': 'efficient_optimized',
        'config_version': '1.0'
    }

def get_backbone_options() -> List[Tuple[str, str]]:
    """Options untuk backbone dropdown"""
    return [
        ('EfficientNet-B4 (Rekomendasi)', 'efficientnet_b4'),
        ('CSPDarknet-S (Baseline)', 'cspdarknet_s')
    ]

def get_model_type_options() -> List[Tuple[str, str]]:
    """Options untuk model type dropdown"""
    return [
        ('EfficientNet Basic', 'efficient_basic'),
        ('EfficientNet Optimized (Rekomendasi)', 'efficient_optimized'),
        ('EfficientNet Advanced', 'efficient_advanced'),
        ('YOLOv5s (Baseline)', 'yolov5s')
    ]

def get_model_type_mapping() -> Dict[str, Dict[str, Any]]:
    """Mapping model_type ke backbone dan feature settings"""
    return {
        'yolov5s': {
            'backbone': 'cspdarknet_s',
            'use_attention': False,
            'use_residual': False,
            'use_ciou': False,
            'disable_features': True
        },
        'efficient_basic': {
            'backbone': 'efficientnet_b4',
            'use_attention': False,
            'use_residual': False,
            'use_ciou': False,
            'disable_features': False
        },
        'efficient_optimized': {
            'backbone': 'efficientnet_b4',
            'use_attention': True,
            'use_residual': False,
            'use_ciou': False,
            'disable_features': False
        },
        'efficient_advanced': {
            'backbone': 'efficientnet_b4',
            'use_attention': True,
            'use_residual': True,
            'use_ciou': True,
            'disable_features': False
        }
    }

def get_backbone_mapping() -> Dict[str, Dict[str, Any]]:
    """Mapping backbone ke default model_type dan feature settings"""
    return {
        'cspdarknet_s': {
            'default_model_type': 'yolov5s',
            'disable_features': True,
            'use_attention': False,
            'use_residual': False,
            'use_ciou': False
        },
        'efficientnet_b4': {
            'default_model_type': 'efficient_optimized',
            'disable_features': False,
            'use_attention': True,
            'use_residual': False,
            'use_ciou': False
        }
    }