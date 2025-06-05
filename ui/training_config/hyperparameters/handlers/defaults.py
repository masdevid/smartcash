"""
File: smartcash/ui/training_config/hyperparameters/handlers/defaults.py
Deskripsi: Default config values untuk hyperparameter dengan structure yang clean
"""

from typing import Dict, Any


def get_default_hyperparameters_config() -> Dict[str, Any]:
    """Return default hyperparameters config dengan struktur yang sesuai dengan hyperparameters_config.yaml"""
    
    return {
        # Override parameter training dari base_config
        'training': {
            'epochs': 100,  # Override dari base_config (30)
            'lr': 0.01,     # Override dari base_config (0.001)
            'batch_size': 16,
            'image_size': 640,
            'optimizer': 'Adam',
            'weight_decay': 0.0005,
            'momentum': 0.937
        },
        
        # Parameter penjadwalan (parameter baru)
        'scheduler': {
            'type': 'cosine',
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1
        },
        
        # Parameter regularisasi (parameter baru)
        'regularization': {
            'augment': True,
            'dropout': 0.0
        },
        
        # Parameter loss (parameter baru)
        'loss': {
            'box_loss_gain': 0.05,
            'cls_loss_gain': 0.5,
            'obj_loss_gain': 1.0
        },
        
        # Parameter anchor (parameter baru)
        'anchor': {
            'anchor_t': 4.0,
            'fl_gamma': 0.0
        },
        
        # Override parameter early stopping
        'early_stopping': {
            'enabled': True,  # Parameter baru
            'patience': 15,  # Override dari base_config (10)
            'min_delta': 0.001  # Parameter baru
        },
        
        # Parameter save best model (parameter baru)
        'save_best': {
            'enabled': True,
            'metric': 'mAP_0.5'
        },
        
        # Metadata
        'config_version': '1.0',
        'description': 'Default hyperparameter configuration untuk SmartCash detection'
    }