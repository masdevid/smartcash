"""
File: smartcash/ui/training_config/training_strategy/handlers/ui_updater.py
Deskripsi: Update UI dari config dengan one-liner style yang DRY
"""

from typing import Dict, Any


def update_training_strategy_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config dengan one-liner style"""
    
    # One-liner safe setter
    set_val = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Extract nested configs dengan fallbacks
    validation = config.get('validation', {})
    multi_scale = config.get('multi_scale', {})
    training_utils = config.get('training_utils', {})
    
    # Update komponen validation UI
    set_val('validation_enabled_checkbox', validation.get('enabled', True))
    set_val('validation_frequency_slider', validation.get('frequency', 1))
    set_val('iou_threshold_slider', validation.get('iou_threshold', 0.6))
    set_val('conf_threshold_slider', validation.get('conf_threshold', 0.001))
    set_val('max_detections_slider', validation.get('max_detections', 300))
    set_val('save_val_results_checkbox', validation.get('save_val_results', True))
    set_val('val_results_dir_text', validation.get('val_results_dir', '/content/runs/val/results'))
    set_val('visualize_val_results_checkbox', validation.get('visualize_val_results', True))
    
    # Update komponen multi-scale UI
    set_val('multi_scale_checkbox', multi_scale.get('enabled', True))
    set_val('min_scale_slider', multi_scale.get('min_scale', 0.5))
    set_val('max_scale_slider', multi_scale.get('max_scale', 1.5))
    set_val('multi_scale_frequency_slider', multi_scale.get('frequency', 10))
    set_val('apply_to_val_checkbox', multi_scale.get('apply_to_val', False))
    
    # Update komponen training utils UI
    set_val('experiment_name_text', training_utils.get('experiment_name', 'efficientnet_b4_train'))
    set_val('checkpoint_dir_text', training_utils.get('checkpoint_dir', '/content/runs/train/checkpoints'))
    set_val('tensorboard_dir_text', training_utils.get('tensorboard_dir', '/content/runs/train/tensorboard'))
    set_val('log_metrics_every_slider', training_utils.get('log_metrics_every', 10))
    set_val('visualize_batch_every_slider', training_utils.get('visualize_batch_every', 100))
    set_val('device_dropdown', training_utils.get('device', 'cuda'))
    set_val('num_workers_slider', training_utils.get('num_workers', 4))
    set_val('mixed_precision_checkbox', training_utils.get('mixed_precision', True))
    set_val('gradient_clipping_slider', training_utils.get('gradient_clipping', 1.0))
    set_val('sync_bn_checkbox', training_utils.get('sync_bn', False))
    set_val('advanced_augmentation_checkbox', training_utils.get('advanced_augmentation', False))