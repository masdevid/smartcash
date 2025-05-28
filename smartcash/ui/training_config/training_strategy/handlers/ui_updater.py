"""
File: smartcash/ui/training_config/training_strategy/handlers/ui_updater.py
Deskripsi: Update UI dari config dengan one-liner style yang DRY
"""

from typing import Dict, Any


def update_training_strategy_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config dengan one-liner style"""
    
    ts_config = config.get('training_strategy', {})
    
    # One-liner safe setter
    set_val = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Extract nested configs dengan fallbacks
    optimizer = ts_config.get('optimizer', {})
    scheduler = ts_config.get('scheduler', {})
    early_stopping = ts_config.get('early_stopping', {})
    checkpoint = ts_config.get('checkpoint', {})
    utils = ts_config.get('utils', {})
    validation = ts_config.get('validation', {})
    multiscale = ts_config.get('multiscale', {})
    
    # Update UI dengan one-liner style
    [set_val(key, value) for key, value in {
        # Parameter utama
        'enabled_checkbox': ts_config.get('enabled', True),
        'batch_size_slider': ts_config.get('batch_size', 16),
        'epochs_slider': ts_config.get('epochs', 100),
        'learning_rate_slider': ts_config.get('learning_rate', 0.001),
        
        # Optimizer
        'optimizer_dropdown': optimizer.get('type', 'adam'),
        'weight_decay_slider': optimizer.get('weight_decay', 0.0005),
        'momentum_slider': optimizer.get('momentum', 0.9),
        
        # Scheduler
        'scheduler_checkbox': scheduler.get('enabled', True),
        'scheduler_dropdown': scheduler.get('type', 'cosine'),
        'warmup_epochs_slider': scheduler.get('warmup_epochs', 5),
        'min_lr_slider': scheduler.get('min_lr', 0.00001),
        
        # Early Stopping
        'early_stopping_checkbox': early_stopping.get('enabled', True),
        'patience_slider': early_stopping.get('patience', 10),
        'min_delta_slider': early_stopping.get('min_delta', 0.001),
        
        # Checkpoint
        'checkpoint_checkbox': checkpoint.get('enabled', True),
        'save_best_only_checkbox': checkpoint.get('save_best_only', True),
        'save_freq_slider': checkpoint.get('save_freq', 1),
        
        # Utils
        'experiment_name': utils.get('experiment_name', 'efficientnet_b4_training'),
        'checkpoint_dir': utils.get('checkpoint_dir', '/content/runs/train/checkpoints'),
        'tensorboard': utils.get('tensorboard', True),
        'log_metrics_every': utils.get('log_metrics_every', 10),
        'visualize_batch_every': utils.get('visualize_batch_every', 100),
        'gradient_clipping': utils.get('gradient_clipping', 1.0),
        'mixed_precision': utils.get('mixed_precision', True),
        'layer_mode': utils.get('layer_mode', 'single'),
        
        # Validation
        'validation_frequency': validation.get('validation_frequency', 1),
        'iou_threshold': validation.get('iou_threshold', 0.6),
        'conf_threshold': validation.get('conf_threshold', 0.001),
        
        # Multiscale
        'multi_scale': multiscale.get('enabled', True)
    }.items()]