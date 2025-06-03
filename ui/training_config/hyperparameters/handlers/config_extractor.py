"""
File: smartcash/ui/training_config/hyperparameters/handlers/config_extractor.py
Deskripsi: Extract konfigurasi hyperparameter dari UI components dengan one-liner style
"""

from typing import Dict, Any
import datetime
from smartcash.ui.training_config.hyperparameters.handlers.defaults import get_default_hyperparameters_config


def extract_hyperparameters_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI components dengan one-liner style"""
    
    # One-liner safe getter
    get_val = lambda key, default: getattr(ui_components.get(key), 'value', default) if key in ui_components else default
    
    # Parameter dasar
    config = {
        'batch_size': get_val('batch_size_slider', 16),
        'image_size': get_val('image_size_slider', 640),
        'epochs': get_val('epochs_slider', 100),
        'dropout': get_val('dropout_slider', 0.0),
        
        # Parameter optimasi
        'optimizer': get_val('optimizer_dropdown', 'Adam'),
        'learning_rate': get_val('learning_rate_slider', 0.01),
        'weight_decay': get_val('weight_decay_slider', 0.0005),
        'momentum': get_val('momentum_slider', 0.937),
        
        # Parameter penjadwalan
        'scheduler': get_val('scheduler_dropdown', 'cosine'),
        'warmup_epochs': get_val('warmup_epochs_slider', 3),
        'warmup_momentum': 0.8,  # Default, tidak ada UI
        'warmup_bias_lr': 0.1,   # Default, tidak ada UI
        
        # Parameter regularisasi
        'augment': get_val('augment_checkbox', True),
        
        # Parameter loss
        'box_loss_gain': get_val('box_loss_gain_slider', 0.05),
        'cls_loss_gain': get_val('cls_loss_gain_slider', 0.5),
        'obj_loss_gain': get_val('obj_loss_gain_slider', 1.0),
        
        # Parameter anchor
        'anchor_t': 4.0,  # Default, tidak ada UI
        'fl_gamma': 0.0,  # Default, tidak ada UI
        
        # Parameter early stopping
        'early_stopping': {
            'enabled': get_val('early_stopping_checkbox', True),
            'patience': get_val('patience_slider', 15),
            'min_delta': get_val('min_delta_slider', 0.001)
        },
        
        # Parameter save best model
        'save_best': {
            'enabled': get_val('save_best_checkbox', True),
            'metric': get_val('checkpoint_metric_dropdown', 'mAP_0.5')
        },
        
        # Metadata
        'config_version': '1.0',
        'updated_at': datetime.datetime.now().isoformat()
    }
    
    return config