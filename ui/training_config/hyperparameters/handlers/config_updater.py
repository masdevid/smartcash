"""
File: smartcash/ui/training_config/hyperparameters/handlers/config_updater.py
Deskripsi: Update UI components dari konfigurasi hyperparameter dengan one-liner style
"""

from typing import Dict, Any


def update_hyperparameters_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI dari config dengan one-liner assignments"""
    
    # One-liner safe setter
    set_val = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Parameter dasar
    set_val('batch_size_slider', config.get('batch_size', 16))
    set_val('image_size_slider', config.get('image_size', 640))
    set_val('epochs_slider', config.get('epochs', 100))
    set_val('dropout_slider', config.get('dropout', 0.0))
    
    # Parameter optimasi
    set_val('optimizer_dropdown', config.get('optimizer', 'Adam'))
    set_val('learning_rate_slider', config.get('learning_rate', 0.01))
    set_val('weight_decay_slider', config.get('weight_decay', 0.0005))
    set_val('momentum_slider', config.get('momentum', 0.937))
    
    # Parameter penjadwalan
    set_val('scheduler_dropdown', config.get('scheduler', 'cosine'))
    set_val('warmup_epochs_slider', config.get('warmup_epochs', 3))
    
    # Parameter loss
    set_val('box_loss_gain_slider', config.get('box_loss_gain', 0.05))
    set_val('cls_loss_gain_slider', config.get('cls_loss_gain', 0.5))
    set_val('obj_loss_gain_slider', config.get('obj_loss_gain', 1.0))
    
    # Parameter regularisasi
    set_val('augment_checkbox', config.get('augment', True))
    
    # Early stopping parameters
    early_stopping = config.get('early_stopping', {})
    set_val('early_stopping_checkbox', early_stopping.get('enabled', True))
    set_val('patience_slider', early_stopping.get('patience', 15))
    set_val('min_delta_slider', early_stopping.get('min_delta', 0.001))
    
    # Checkpoint parameters
    save_best = config.get('save_best', {})
    set_val('save_best_checkbox', save_best.get('enabled', True))
    set_val('checkpoint_metric_dropdown', save_best.get('metric', 'mAP_0.5'))