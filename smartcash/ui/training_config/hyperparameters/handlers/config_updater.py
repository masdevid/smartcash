"""
File: smartcash/ui/training_config/hyperparameters/handlers/config_updater.py
Deskripsi: Update UI components dari konfigurasi hyperparameter dengan one-liner style
"""

from typing import Dict, Any


def update_hyperparameters_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI dari config dengan one-liner assignments"""
    
    hp_config = config.get('hyperparameters', {})
    
    # Training parameters - one-liner updates
    training = hp_config.get('training', {})
    ui_components.get('batch_size_slider') and setattr(ui_components['batch_size_slider'], 'value', training.get('batch_size', 16))
    ui_components.get('image_size_slider') and setattr(ui_components['image_size_slider'], 'value', training.get('image_size', 640))
    ui_components.get('epochs_slider') and setattr(ui_components['epochs_slider'], 'value', training.get('epochs', 100))
    ui_components.get('dropout_slider') and setattr(ui_components['dropout_slider'], 'value', training.get('dropout', 0.0))
    
    # Optimizer parameters - one-liner updates
    optimizer = hp_config.get('optimizer', {})
    ui_components.get('optimizer_dropdown') and setattr(ui_components['optimizer_dropdown'], 'value', optimizer.get('type', 'SGD'))
    ui_components.get('learning_rate_slider') and setattr(ui_components['learning_rate_slider'], 'value', optimizer.get('learning_rate', 0.01))
    ui_components.get('weight_decay_slider') and setattr(ui_components['weight_decay_slider'], 'value', optimizer.get('weight_decay', 0.0005))
    ui_components.get('momentum_slider') and setattr(ui_components['momentum_slider'], 'value', optimizer.get('momentum', 0.937))
    
    # Scheduler parameters - one-liner updates
    scheduler = hp_config.get('scheduler', {})
    ui_components.get('scheduler_checkbox') and setattr(ui_components['scheduler_checkbox'], 'value', scheduler.get('enabled', True))
    ui_components.get('scheduler_dropdown') and setattr(ui_components['scheduler_dropdown'], 'value', scheduler.get('type', 'cosine'))
    ui_components.get('warmup_epochs_slider') and setattr(ui_components['warmup_epochs_slider'], 'value', scheduler.get('warmup_epochs', 3))
    
    # Loss parameters - one-liner updates
    loss = hp_config.get('loss', {})
    ui_components.get('box_loss_gain_slider') and setattr(ui_components['box_loss_gain_slider'], 'value', loss.get('box_loss_gain', 0.05))
    ui_components.get('cls_loss_gain_slider') and setattr(ui_components['cls_loss_gain_slider'], 'value', loss.get('cls_loss_gain', 0.5))
    ui_components.get('obj_loss_gain_slider') and setattr(ui_components['obj_loss_gain_slider'], 'value', loss.get('obj_loss_gain', 1.0))
    
    # Early stopping parameters - one-liner updates
    early_stopping = hp_config.get('early_stopping', {})
    ui_components.get('early_stopping_checkbox') and setattr(ui_components['early_stopping_checkbox'], 'value', early_stopping.get('enabled', True))
    ui_components.get('patience_slider') and setattr(ui_components['patience_slider'], 'value', early_stopping.get('patience', 10))
    ui_components.get('min_delta_slider') and setattr(ui_components['min_delta_slider'], 'value', early_stopping.get('min_delta', 0.001))
    
    # Augmentation and checkpoint - one-liner updates
    augmentation = hp_config.get('augmentation', {})
    checkpoint = hp_config.get('checkpoint', {})
    ui_components.get('augment_checkbox') and setattr(ui_components['augment_checkbox'], 'value', augmentation.get('enabled', True))
    ui_components.get('save_best_checkbox') and setattr(ui_components['save_best_checkbox'], 'value', checkpoint.get('save_best', True))
    ui_components.get('checkpoint_metric_dropdown') and setattr(ui_components['checkpoint_metric_dropdown'], 'value', checkpoint.get('metric', 'mAP_0.5'))