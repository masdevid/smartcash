"""
File: smartcash/ui/training_config/training_strategy/handlers/defaults.py
Deskripsi: Default values untuk training strategy config yang DRY
"""

from typing import Dict, Any


def get_default_training_strategy_config() -> Dict[str, Any]:
    """Return default training strategy config dengan one-liner structure"""
    return {
        'training_strategy': {
            'enabled': True,
            'batch_size': 16,
            'epochs': 100,
            'learning_rate': 0.001,
            'optimizer': {'type': 'adam', 'weight_decay': 0.0005, 'momentum': 0.9},
            'scheduler': {'enabled': True, 'type': 'cosine', 'warmup_epochs': 5, 'min_lr': 0.00001},
            'early_stopping': {'enabled': True, 'patience': 10, 'min_delta': 0.001},
            'checkpoint': {'enabled': True, 'save_best_only': True, 'save_freq': 1},
            'utils': {
                'experiment_name': 'efficientnet_b4_training',
                'checkpoint_dir': '/content/runs/train/checkpoints',
                'tensorboard': True,
                'log_metrics_every': 10,
                'visualize_batch_every': 100,
                'gradient_clipping': 1.0,
                'mixed_precision': True,
                'layer_mode': 'single'
            },
            'validation': {'validation_frequency': 1, 'iou_threshold': 0.6, 'conf_threshold': 0.001},
            'multiscale': {'enabled': True}
        }
    }