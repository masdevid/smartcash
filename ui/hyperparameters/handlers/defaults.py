"""
File: smartcash/ui/hyperparameters/handlers/defaults.py
Deskripsi: Default hyperparameter configuration dengan struktur yang clean dan fokus pada parameter penting
"""

from typing import Dict, Any


def get_default_hyperparameters_config() -> Dict[str, Any]:
    """Return default hyperparameters config dengan parameter yang penting dan dapat diubah saja"""
    
    return {
        # Inherit dari base_config.yaml
        '_base_': 'base_config.yaml',
        
        # Parameter training utama yang sering diubah
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.01,
            'image_size': 640,
            'mixed_precision': True,
            'gradient_accumulation': 1,
            'gradient_clipping': 1.0
        },
        
        # Parameter optimizer yang penting
        'optimizer': {
            'type': 'SGD',
            'weight_decay': 0.0005,
            'momentum': 0.937
        },
        
        # Parameter scheduler
        'scheduler': {
            'type': 'cosine',
            'warmup_epochs': 3
        },
        
        # Parameter loss weights
        'loss': {
            'box_loss_gain': 0.05,
            'cls_loss_gain': 0.5,
            'obj_loss_gain': 1.0
        },
        
        # Early stopping configuration
        'early_stopping': {
            'enabled': True,
            'patience': 15,
            'min_delta': 0.001
        },
        
        # Checkpoint configuration
        'checkpoint': {
            'save_best': True,
            'metric': 'mAP_0.5'
        },
        
        # Metadata
        'config_version': '1.0',
        'description': 'Default hyperparameter configuration untuk SmartCash detection',
        'module_name': 'hyperparameters'
    }


def get_optimizer_options() -> list:
    """Return available optimizer options"""
    return ['SGD', 'Adam', 'AdamW']


def get_scheduler_options() -> list:
    """Return available scheduler options"""
    return ['cosine', 'step', 'exponential', 'none']


def get_checkpoint_metric_options() -> list:
    """Return available checkpoint metric options"""
    return ['mAP_0.5', 'mAP_0.5:0.95', 'precision', 'recall', 'f1', 'loss']