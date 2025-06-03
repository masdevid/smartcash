"""
File: smartcash/ui/training_config/training_strategy/handlers/defaults.py
Deskripsi: Default values untuk training strategy config yang DRY
"""

from typing import Dict, Any


def get_default_training_strategy_config() -> Dict[str, Any]:
    """Return default training strategy config dengan one-liner structure"""
    return {
        # Parameter validasi
        'validation': {
            'frequency': 1,
            'iou_thres': 0.6,
            'conf_thres': 0.001
        },
        
        # Parameter multi-scale training
        'multi_scale': True,
        
        # Konfigurasi tambahan untuk proses training
        'training_utils': {
            'experiment_name': 'efficientnet_b4_training',
            'checkpoint_dir': '/content/runs/train/checkpoints',
            'tensorboard': True,
            'log_metrics_every': 10,  # Log metrik setiap 10 batch
            'visualize_batch_every': 100,  # Visualisasi batch setiap 100 batch
            'gradient_clipping': 1.0,  # Clipping gradien maksimum
            'mixed_precision': True,  # Gunakan mixed precision training
            'layer_mode': 'single'  # Opsi: 'single' atau 'multilayer'
        },
        
        # Metadata
        'config_version': '1.0',
        'description': 'Default training strategy configuration untuk SmartCash detection'
    }