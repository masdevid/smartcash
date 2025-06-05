"""
File: smartcash/ui/training_config/strategy/handlers/config_extractor.py
Deskripsi: Extract config dari UI dengan one-liner style yang DRY
"""

from typing import Dict, Any
import datetime


def extract_strategy_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI components dengan one-liner style sesuai structure di training_config.yaml"""
    
    # One-liner safe getter
    get_val = lambda key, default: getattr(ui_components.get(key), 'value', default) if key in ui_components else default
    
    # Metadata untuk config yang diperbarui
    current_time = datetime.datetime.now().isoformat()
    
    # Buat config sesuai dengan training_config.yaml yang diperbarui
    config = {
        # Inherit dari base_config.yaml dan config lainnya
        '_base_': ['base_config.yaml', 'hyperparameters_config.yaml', 'model_config.yaml'],
        
        # Konfigurasi validasi
        'validation': {
            'enabled': get_val('validation_enabled_checkbox', True),
            'frequency': get_val('validation_frequency_slider', 1),
            'iou_thres': get_val('iou_threshold_slider', 0.6),
            'conf_thres': get_val('conf_threshold_slider', 0.001),
            'max_detections': 100,  # Default, tidak ada UI
            'save_val_predictions': get_val('save_val_predictions_checkbox', True),
            'val_img_count': get_val('val_img_count_slider', 100)
        },
        
        # Multi-scale training
        'multi_scale': {
            'enabled': get_val('multi_scale_checkbox', True),
            'img_size_min': get_val('img_size_min_slider', 320),
            'img_size_max': get_val('img_size_max_slider', 640),
            'step_size': 32  # Default, tidak ada UI
        },
        
        # Training utilities
        'training_utils': {
            # Experiment tracking
            'experiment_name': get_val('experiment_name_text', 'efficientnet_b4_training'),
            'checkpoint_dir': get_val('checkpoint_dir_text', '/content/runs/train/checkpoints'),
            'tensorboard': get_val('tensorboard_checkbox', True),
            'log_metrics_every': get_val('log_metrics_every_slider', 10),
            'visualize_batch_every': get_val('visualize_batch_every_slider', 100),
            
            # Training optimizations
            'gradient_clipping': get_val('gradient_clipping_slider', 1.0),
            'mixed_precision': get_val('mixed_precision_checkbox', True),
            'layer_mode': get_val('layer_mode_dropdown', 'single'),
            
            # Callbacks dan hooks
            'callbacks': {
                'early_stopping': get_val('early_stopping_callback_checkbox', True),
                'model_pruning': get_val('model_pruning_checkbox', False),
                'learning_rate_finder': get_val('lr_finder_checkbox', False)
            },
            
            # Distributed training
            'distributed': {
                'enabled': get_val('distributed_checkbox', False),
                'backend': 'nccl',  # Default, tidak ada UI
                'sync_bn': False    # Default, tidak ada UI
            }
        },
        
        # Metadata
        'config_version': '1.0',
        'updated_at': current_time
    }
    
    # Logging untuk debugging
    if 'status_panel' in ui_components:
        from smartcash.ui.utils.alert_utils import update_status_panel
        update_status_panel(ui_components['status_panel'], f"📊 Konfigurasi strategi training berhasil diekstrak", "info")
    
    return config