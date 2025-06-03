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
    
    # Import default configs
    from smartcash.ui.training_config.backbone.handlers.defaults import get_default_backbone_config
    from smartcash.ui.training_config.hyperparameters.handlers.defaults import get_default_hyperparameters_config
    
    # Get backbone and hyperparameters defaults to use in training strategy
    backbone_defaults = get_default_backbone_config()
    hyperparameters_defaults = get_default_hyperparameters_config()
    
    # Struktur config sesuai dengan training_config.yaml
    return {
        # Parameter validasi
        'validation': {
            'enabled': get_val('validation_enabled_checkbox', True),
            'frequency': get_val('validation_frequency_slider', 1),
            'iou_threshold': get_val('iou_threshold_slider', 0.6),
            'conf_threshold': get_val('conf_threshold_slider', 0.001),
            'max_detections': get_val('max_detections_slider', 300),
            'save_val_results': get_val('save_val_results_checkbox', True),
            'val_results_dir': get_val('val_results_dir_text', '/content/runs/val/results'),
            'visualize_val_results': get_val('visualize_val_results_checkbox', True)
        },
        
        # Parameter multi-scale training
        'multi_scale': {
            'enabled': get_val('multi_scale_checkbox', True),
            'min_scale': get_val('min_scale_slider', 0.5),
            'max_scale': get_val('max_scale_slider', 1.5),
            'frequency': get_val('multi_scale_frequency_slider', 10),
            'apply_to_val': get_val('apply_to_val_checkbox', False)
        },
        
        # Parameter utilitas training
        'training_utils': {
            'experiment_name': get_val('experiment_name_text', 'efficientnet_b4_train'),
            'checkpoint_dir': get_val('checkpoint_dir_text', '/content/runs/train/checkpoints'),
            'tensorboard_dir': get_val('tensorboard_dir_text', '/content/runs/train/tensorboard'),
            'log_metrics_every': get_val('log_metrics_every_slider', 10),
            'visualize_batch_every': get_val('visualize_batch_every_slider', 100),
            'device': get_val('device_dropdown', 'cuda'),
            'num_workers': get_val('num_workers_slider', 4),
            'mixed_precision': get_val('mixed_precision_checkbox', True),
            'gradient_clipping': get_val('gradient_clipping_slider', 1.0),
            'sync_bn': get_val('sync_bn_checkbox', False),
            'advanced_augmentation': get_val('advanced_augmentation_checkbox', False)
        },
        
        # Metadata
        'config_version': '1.0',
        'updated_at': datetime.datetime.now().isoformat(),
        
        # Reference ke backbone dan hyperparameters config
        '_base_': {
            'model': backbone_defaults,
            'hyperparameters': hyperparameters_defaults
        }
    }