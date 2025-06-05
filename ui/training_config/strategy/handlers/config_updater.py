"""
File: smartcash/ui/training_config/strategy/handlers/config_updater.py
Deskripsi: Update UI dari config dengan one-liner style yang DRY sesuai struktur YAML
"""

from typing import Dict, Any


def update_strategy_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config dengan one-liner style sesuai struktur training_config.yaml yang diperbarui"""
    
    # One-liner safe setter untuk memastikan keamanan akses UI components
    set_val = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Extract nested configs dengan fallbacks
    validation_config = config.get('validation', {})
    multi_scale_config = config.get('multi_scale', {})
    training_utils_config = config.get('training_utils', {})
    
    # Extract nested configs dalam training_utils
    callbacks_config = training_utils_config.get('callbacks', {})
    distributed_config = training_utils_config.get('distributed', {})
    
    # Update UI components dengan mapping approach
    field_mappings = [
        # Validation parameters
        ('validation_enabled_checkbox', validation_config, 'enabled', True),
        ('validation_frequency_slider', validation_config, 'frequency', 1),
        ('iou_threshold_slider', validation_config, 'iou_thres', 0.6),
        ('conf_threshold_slider', validation_config, 'conf_thres', 0.001),
        ('save_val_predictions_checkbox', validation_config, 'save_val_predictions', True),
        ('val_img_count_slider', validation_config, 'val_img_count', 100),
        
        # Multi-scale parameters
        ('multi_scale_checkbox', multi_scale_config, 'enabled', True),
        ('img_size_min_slider', multi_scale_config, 'img_size_min', 320),
        ('img_size_max_slider', multi_scale_config, 'img_size_max', 640),
        
        # Training utils - experiment tracking
        ('experiment_name_text', training_utils_config, 'experiment_name', 'efficientnet_b4_training'),
        ('checkpoint_dir_text', training_utils_config, 'checkpoint_dir', '/content/runs/train/checkpoints'),
        ('tensorboard_checkbox', training_utils_config, 'tensorboard', True),
        ('log_metrics_every_slider', training_utils_config, 'log_metrics_every', 10),
        ('visualize_batch_every_slider', training_utils_config, 'visualize_batch_every', 100),
        
        # Training utils - optimizations
        ('gradient_clipping_slider', training_utils_config, 'gradient_clipping', 1.0),
        ('mixed_precision_checkbox', training_utils_config, 'mixed_precision', True),
        ('layer_mode_dropdown', training_utils_config, 'layer_mode', 'single'),
        
        # Training utils - callbacks
        ('early_stopping_callback_checkbox', callbacks_config, 'early_stopping', True),
        ('model_pruning_checkbox', callbacks_config, 'model_pruning', False),
        ('lr_finder_checkbox', callbacks_config, 'learning_rate_finder', False),
        
        # Training utils - distributed
        ('distributed_checkbox', distributed_config, 'enabled', False)
    ]
    
    # Apply all updates dengan one-liner
    [set_val(component_key, source_config.get(config_key, default_value)) 
     for component_key, source_config, config_key, default_value in field_mappings 
     if component_key in ui_components]
    
    # Logging untuk informasi
    if 'status_panel' in ui_components:
        from smartcash.ui.utils.alert_utils import update_status_panel
        update_status_panel(ui_components['status_panel'], f"📊 UI training strategy berhasil diperbarui", "info")