"""
File: smartcash/ui/training_config/hyperparameters/handlers/config_extractor.py
Deskripsi: Extract konfigurasi hyperparameter dari UI components dengan one-liner style
"""

from typing import Dict, Any
import datetime
from smartcash.ui.training_config.hyperparameters.handlers.defaults import get_default_hyperparameters_config


def extract_hyperparameters_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI components dengan one-liner style sesuai struktur hyperparameters_config.yaml"""
    
    # One-liner safe getter
    get_val = lambda key, default: getattr(ui_components.get(key), 'value', default) if key in ui_components else default
    
    # Metadata untuk config yang diperbarui
    current_time = datetime.datetime.now().isoformat()
    
    # Buat config sesuai dengan hyperparameters_config.yaml yang diperbarui
    return {
        # Inherit dari base_config.yaml
        '_base_': 'base_config.yaml',
        
        # Parameter training dasar
        'training': {
            'batch_size': get_val('batch_size_slider', 16),
            'image_size': get_val('image_size_slider', 640),
            'epochs': get_val('epochs_slider', 100),
            'dropout': get_val('dropout_slider', 0.0),
            'workers': 4,  # Default, tidak ada UI
            'device': '',   # Auto-detect
            'multi_gpu': False,  # Default, tidak ada UI
            'mixed_precision': get_val('mixed_precision_checkbox', True),
            'gradient_accumulation': get_val('gradient_accumulation_slider', 1),
            'gradient_clipping': get_val('gradient_clipping_slider', 0.0),
            'sync_bn': False  # Default, tidak ada UI
        },
        
        # Parameter optimasi
        'optimizer': {
            'type': get_val('optimizer_dropdown', 'Adam'),
            'learning_rate': get_val('learning_rate_slider', 0.01),
            'weight_decay': get_val('weight_decay_slider', 0.0005),
            'momentum': get_val('momentum_slider', 0.937),
            'nesterov': True  # Default, tidak ada UI
        },
        
        # Parameter penjadwalan
        'scheduler': {
            'type': get_val('scheduler_dropdown', 'cosine'),
            'warmup_epochs': get_val('warmup_epochs_slider', 3),
            'warmup_momentum': 0.8,  # Default, tidak ada UI
            'warmup_bias_lr': 0.1,   # Default, tidak ada UI
            'final_lr_factor': 0.01,  # Default, tidak ada UI
            'patience': 3  # Default, tidak ada UI untuk ReduceLROnPlateau
        },
        
        # Parameter regularisasi
        'regularization': {
            'augment': get_val('augment_checkbox', True),
            'label_smoothing': get_val('label_smoothing_slider', 0.0),
            'dropout': get_val('dropout_slider', 0.0),
            'weight_decay': get_val('weight_decay_slider', 0.0005)
        },
        
        # Parameter loss
        'loss': {
            'box_loss_gain': get_val('box_loss_gain_slider', 0.05),
            'cls_loss_gain': get_val('cls_loss_gain_slider', 0.5),
            'obj_loss_gain': get_val('obj_loss_gain_slider', 1.0),
            'cls_positive_weight': 1.0,  # Default, tidak ada UI
            'obj_positive_weight': 1.0   # Default, tidak ada UI
        },
        
        # Parameter anchor
        'anchor': {
            'anchor_t': 4.0,  # Default, tidak ada UI
            'fl_gamma': 0.0,  # Default, tidak ada UI
            'auto_anchor': True  # Default, tidak ada UI
        },
        
        # Parameter early stopping
        'early_stopping': {
            'enabled': get_val('early_stopping_checkbox', True),
            'patience': get_val('patience_slider', 15),
            'min_delta': get_val('min_delta_slider', 0.001),
            'metric': 'mAP_0.5',  # Default, tidak ada UI
            'mode': 'max'  # Default, tidak ada UI
        },
        
        # Parameter save best model
        'save_best': {
            'enabled': get_val('save_best_checkbox', True),
            'metric': get_val('checkpoint_metric_dropdown', 'mAP_0.5'),
            'mode': 'max',  # Default, tidak ada UI
            'every_epoch': False  # Default, tidak ada UI
        },
        
        # Metadata
        'config_version': '1.0',
        'updated_at': current_time
    }