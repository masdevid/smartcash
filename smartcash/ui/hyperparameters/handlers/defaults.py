"""
File: smartcash/ui/hyperparameters/handlers/defaults.py
Deskripsi: Default hyperparameter config essentials untuk backend model
"""

from typing import Dict, Any


def get_default_hyperparameters_config() -> Dict[str, Any]:
    """Return default hyperparameters config dengan parameter backend essentials saja 🎯"""
    
    return {
        # Inherit dari base_config.yaml
        '_base_': 'base_config.yaml',
        
        # Parameter training utama
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.01,
            'image_size': 640
        },
        
        # Parameter optimizer essentials
        'optimizer': {
            'type': 'SGD',
            'weight_decay': 0.0005,
            'momentum': 0.937  # Backend default
        },
        
        # Parameter scheduler
        'scheduler': {
            'type': 'cosine',
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,  # Backend default
            'warmup_bias_lr': 0.1    # Backend default
        },
        
        # Parameter loss weights
        'loss': {
            'box_loss_gain': 0.05,
            'cls_loss_gain': 0.5,
            'obj_loss_gain': 1.0
        },
        
        # Early stopping essentials
        'early_stopping': {
            'enabled': True,
            'patience': 15,
            'min_delta': 0.001  # Backend default
        },
        
        # Checkpoint essentials
        'checkpoint': {
            'save_best': True,
            'metric': 'mAP_0.5'
        },
        
        # Backend defaults tidak perlu di UI
        'regularization': {
            'dropout': 0.0
        },
        
        'anchor': {
            'anchor_t': 4.0,
            'fl_gamma': 0.0
        },
        
        # Metadata
        'config_version': '2.0',
        'description': 'Simplified hyperparameter config untuk backend model',
        'module_name': 'hyperparameters'
    }


def get_optimizer_options() -> list:
    """Return available optimizer options untuk backend"""
    return ['SGD', 'Adam', 'AdamW']


def get_scheduler_options() -> list:
    """Return available scheduler options untuk backend"""
    return ['cosine', 'step', 'exponential', 'none']


def get_checkpoint_metric_options() -> list:
    """Return available checkpoint metric options untuk backend"""
    return ['mAP_0.5', 'mAP_0.5:0.95', 'precision', 'recall', 'f1', 'loss']


# Removed unused functions dan parameters:
# - Mixed precision options (tidak digunakan backend)
# - Gradient accumulation options (tidak di backend)
# - Advanced scheduler options (tidak diperlukan UI)