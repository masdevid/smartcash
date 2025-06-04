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
    
    # Current time untuk metadata
    current_time = datetime.datetime.now().isoformat()
    
    # Struktur config yang sesuai dengan training_config.yaml
    config = {
        # Inherit dari konfigurasi lain
        '_base_': ['base_config.yaml', 'hyperparameters_config.yaml', 'model_config.yaml'],
        
        # Parameter validasi sesuai training_config.yaml
        'validation': {
            'frequency': get_val('validation_frequency_slider', 1),
            'iou_thres': get_val('iou_threshold_slider', 0.6),  # Nama field sesuai YAML
            'conf_thres': get_val('conf_threshold_slider', 0.001)  # Nama field sesuai YAML
        },
        
        # Multi-scale sebagai boolean
        'multi_scale': get_val('multi_scale_checkbox', True),
        
        # Konfigurasi tambahan untuk proses training
        'training_utils': {
            'experiment_name': get_val('experiment_name_text', 'efficientnet_b4_training'),
            'checkpoint_dir': get_val('checkpoint_dir_text', '/content/runs/train/checkpoints'),
            'tensorboard': get_val('tensorboard_checkbox', True),
            'log_metrics_every': get_val('log_metrics_every_slider', 10),
            'visualize_batch_every': get_val('visualize_batch_every_slider', 100),
            'gradient_clipping': get_val('gradient_clipping_slider', 1.0),
            'mixed_precision': get_val('mixed_precision_checkbox', True),
            'layer_mode': get_val('layer_mode_dropdown', 'single')
        },
        
        # Metadata untuk tracking
        'config_version': '1.0',
        'updated_at': current_time
    }
    
    # Logging untuk debugging
    if 'status_panel' in ui_components:
        from smartcash.ui.utils.alert_utils import update_status_panel
        update_status_panel(ui_components['status_panel'], f"📊 Konfigurasi strategi training berhasil diekstrak", "info")
    
    return config