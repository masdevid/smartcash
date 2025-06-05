"""
File: smartcash/ui/training_config/strategy/handlers/defaults.py
Deskripsi: Default values untuk training strategy config yang DRY
"""

from typing import Dict, Any


def get_default_strategy_config() -> Dict[str, Any]:
    """Return default training strategy config dengan struktur yang sesuai dengan training_config.yaml"""
    return {
        # Parameter validasi (parameter baru)
        'validation': {
            'frequency': 1,
            'iou_thres': 0.6,
            'conf_thres': 0.001
        },
        
        # Parameter multi-scale training (parameter baru)
        'multi_scale': True,
        
        # Override konfigurasi training_utils dari base_config
        'training_utils': {
            'experiment_name': 'efficientnet_b4_training',  # Override dari base_config (training)
            'checkpoint_dir': '/content/runs/train/checkpoints',  # Override dari base_config (runs/train/checkpoints)
            'tensorboard': True,  # Override dari base_config (false)
            'log_metrics_every': 10,  # Override dari base_config
            'visualize_batch_every': 100,  # Override dari base_config (0)
            'gradient_clipping': 1.0,  # Override dari base_config (0.0)
            'mixed_precision': True,  # Override dari base_config (false)
            'layer_mode': 'single'  # Parameter baru
        },
        
        # Referensi ke konfigurasi yang diwarisi
        'inherited_configs': {
            'hyperparameters': 'hyperparameters_config.yaml',
            'model': 'model_config.yaml'
        },
        
        # Metadata
        'config_version': '1.0',
        'description': 'Default training strategy configuration untuk SmartCash detection'
    }