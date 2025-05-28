"""
File: smartcash/ui/training_config/hyperparameters/handlers/config_extractor.py
Deskripsi: Extract konfigurasi hyperparameter dari UI components dengan one-liner style
"""

from typing import Dict, Any
from smartcash.ui.training_config.hyperparameters.handlers.defaults import get_default_hyperparameters_config


def extract_hyperparameters_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI components dengan one-liner style"""
    
    # Base config structure
    config = {'hyperparameters': {}}
    hp = config['hyperparameters']
    
    # Training parameters - one-liner extraction
    hp['training'] = {
        'batch_size': ui_components.get('batch_size_slider', type('obj', (), {'value': 16})).value,
        'image_size': ui_components.get('image_size_slider', type('obj', (), {'value': 640})).value,
        'epochs': ui_components.get('epochs_slider', type('obj', (), {'value': 100})).value,
        'dropout': ui_components.get('dropout_slider', type('obj', (), {'value': 0.0})).value
    }
    
    # Optimizer parameters - one-liner extraction
    hp['optimizer'] = {
        'type': ui_components.get('optimizer_dropdown', type('obj', (), {'value': 'SGD'})).value,
        'learning_rate': ui_components.get('learning_rate_slider', type('obj', (), {'value': 0.01})).value,
        'weight_decay': ui_components.get('weight_decay_slider', type('obj', (), {'value': 0.0005})).value,
        'momentum': ui_components.get('momentum_slider', type('obj', (), {'value': 0.937})).value
    }
    
    # Scheduler parameters - one-liner extraction
    hp['scheduler'] = {
        'enabled': ui_components.get('scheduler_checkbox', type('obj', (), {'value': True})).value,
        'type': ui_components.get('scheduler_dropdown', type('obj', (), {'value': 'cosine'})).value,
        'warmup_epochs': ui_components.get('warmup_epochs_slider', type('obj', (), {'value': 3})).value
    }
    
    # Loss parameters - one-liner extraction
    hp['loss'] = {
        'box_loss_gain': ui_components.get('box_loss_gain_slider', type('obj', (), {'value': 0.05})).value,
        'cls_loss_gain': ui_components.get('cls_loss_gain_slider', type('obj', (), {'value': 0.5})).value,
        'obj_loss_gain': ui_components.get('obj_loss_gain_slider', type('obj', (), {'value': 1.0})).value
    }
    
    # Early stopping parameters - one-liner extraction
    hp['early_stopping'] = {
        'enabled': ui_components.get('early_stopping_checkbox', type('obj', (), {'value': True})).value,
        'patience': ui_components.get('patience_slider', type('obj', (), {'value': 10})).value,
        'min_delta': ui_components.get('min_delta_slider', type('obj', (), {'value': 0.001})).value
    }
    
    # Augmentation and checkpoint - one-liner extraction  
    hp['augmentation'] = {'enabled': ui_components.get('augment_checkbox', type('obj', (), {'value': True})).value}
    hp['checkpoint'] = {
        'save_best': ui_components.get('save_best_checkbox', type('obj', (), {'value': True})).value,
        'metric': ui_components.get('checkpoint_metric_dropdown', type('obj', (), {'value': 'mAP_0.5'})).value
    }
    
    return config