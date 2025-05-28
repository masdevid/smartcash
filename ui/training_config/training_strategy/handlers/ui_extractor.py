"""
File: smartcash/ui/training_config/training_strategy/handlers/ui_extractor.py
Deskripsi: Extract config dari UI dengan one-liner style yang DRY
"""

from typing import Dict, Any
import datetime


def extract_training_strategy_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI components dengan one-liner style"""
    
    # One-liner safe getter
    get_val = lambda key, default: getattr(ui_components.get(key), 'value', default) if key in ui_components else default
    
    return {
        'training_strategy': {
            'enabled': get_val('enabled_checkbox', True),
            'batch_size': get_val('batch_size_slider', 16),
            'epochs': get_val('epochs_slider', 100),
            'learning_rate': get_val('learning_rate_slider', 0.001),
            
            'optimizer': {
                'type': get_val('optimizer_dropdown', 'adam'),
                'weight_decay': get_val('weight_decay_slider', 0.0005),
                'momentum': get_val('momentum_slider', 0.9)
            },
            
            'scheduler': {
                'enabled': get_val('scheduler_checkbox', True),
                'type': get_val('scheduler_dropdown', 'cosine'),
                'warmup_epochs': get_val('warmup_epochs_slider', 5),
                'min_lr': get_val('min_lr_slider', 0.00001)
            },
            
            'early_stopping': {
                'enabled': get_val('early_stopping_checkbox', True),
                'patience': get_val('patience_slider', 10),
                'min_delta': get_val('min_delta_slider', 0.001)
            },
            
            'checkpoint': {
                'enabled': get_val('checkpoint_checkbox', True),
                'save_best_only': get_val('save_best_only_checkbox', True),
                'save_freq': get_val('save_freq_slider', 1)
            },
            
            'utils': {
                'experiment_name': get_val('experiment_name', 'efficientnet_b4_training'),
                'checkpoint_dir': get_val('checkpoint_dir', '/content/runs/train/checkpoints'),
                'tensorboard': get_val('tensorboard', True),
                'log_metrics_every': get_val('log_metrics_every', 10),
                'visualize_batch_every': get_val('visualize_batch_every', 100),
                'gradient_clipping': get_val('gradient_clipping', 1.0),
                'mixed_precision': get_val('mixed_precision', True),
                'layer_mode': get_val('layer_mode', 'single')
            },
            
            'validation': {
                'validation_frequency': get_val('validation_frequency', 1),
                'iou_threshold': get_val('iou_threshold', 0.6),
                'conf_threshold': get_val('conf_threshold', 0.001)
            },
            
            'multiscale': {
                'enabled': get_val('multi_scale', True)
            }
        },
        'timestamp': datetime.datetime.now().isoformat()
    }