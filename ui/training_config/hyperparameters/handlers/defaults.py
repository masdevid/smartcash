"""
File: smartcash/ui/training_config/hyperparameters/handlers/defaults.py
Deskripsi: Default config values untuk hyperparameter dengan structure yang clean
"""

from typing import Dict, Any


def get_default_hyperparameters_config() -> Dict[str, Any]:
    """Return default hyperparameters config dengan one-liner structure"""
    
    return {
        # Parameter dasar
        'batch_size': 16,
        'image_size': 640,
        'epochs': 100,
        
        # Parameter optimasi
        'optimizer': 'Adam',
        'learning_rate': 0.01,
        'weight_decay': 0.0005,
        'momentum': 0.937,
        
        # Parameter penjadwalan
        'scheduler': 'cosine',
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Parameter regularisasi
        'augment': True,
        'dropout': 0.0,
        
        # Parameter loss
        'box_loss_gain': 0.05,
        'cls_loss_gain': 0.5,
        'obj_loss_gain': 1.0,
        
        # Parameter anchor
        'anchor_t': 4.0,
        'fl_gamma': 0.0,
        
        # Parameter early stopping
        'early_stopping': {
            'enabled': True,
            'patience': 15,
            'min_delta': 0.001
        },
        
        # Parameter save best model
        'save_best': {
            'enabled': True,
            'metric': 'mAP_0.5'
        },
        
        # Metadata
        'config_version': '1.0',
        'description': 'Default hyperparameter configuration untuk SmartCash detection'
    }