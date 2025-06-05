"""
File: smartcash/ui/strategy/handlers/defaults.py
Deskripsi: Default values untuk strategy (bukan hyperparameters)
"""

from typing import Dict, Any


def get_default_strategy_config() -> Dict[str, Any]:
    """Default strategy config fokus pada non-hyperparameters"""
    return {
        # Validation strategy
        'validation': {
            'frequency': 1,
            'iou_thres': 0.6,
            'conf_thres': 0.001,
            'max_detections': 300
        },
        
        # Training utilities
        'training_utils': {
            'experiment_name': 'efficientnet_b4_training',
            'checkpoint_dir': '/content/runs/train/checkpoints',
            'tensorboard': True,
            'log_metrics_every': 10,
            'visualize_batch_every': 100,
            'gradient_clipping': 1.0,
            'layer_mode': 'single'
        },
        
        # Multi-scale training
        'multi_scale': {
            'enabled': True,
            'img_size_min': 320,
            'img_size_max': 640,
            'step_size': 32
        },
        
        # Metadata
        'config_version': '2.0',
        'description': 'Strategy config untuk training utilities dan validation'
    }