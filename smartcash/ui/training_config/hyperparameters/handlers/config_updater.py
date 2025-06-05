"""
File: smartcash/ui/training_config/hyperparameters/handlers/config_updater.py
Deskripsi: Update UI components dari konfigurasi hyperparameter dengan one-liner style
"""

from typing import Dict, Any


def update_hyperparameters_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI dari config dengan one-liner assignments sesuai struktur hyperparameters_config.yaml"""
    
    # Extract nested configs dengan fallbacks
    training_config = config.get('training', {})
    optimizer_config = config.get('optimizer', {})
    scheduler_config = config.get('scheduler', {})
    regularization_config = config.get('regularization', {})
    loss_config = config.get('loss', {})
    early_stopping_config = config.get('early_stopping', {})
    save_best_config = config.get('save_best', {})
    
    # One-liner safe setter
    set_val = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Update UI components dengan mapping approach
    field_mappings = [
        # Parameter training dasar
        ('batch_size_slider', training_config, 'batch_size', 16),
        ('image_size_slider', training_config, 'image_size', 640),
        ('epochs_slider', training_config, 'epochs', 100),
        ('dropout_slider', training_config, 'dropout', 0.0),
        ('mixed_precision_checkbox', training_config, 'mixed_precision', True),
        ('gradient_accumulation_slider', training_config, 'gradient_accumulation', 1),
        ('gradient_clipping_slider', training_config, 'gradient_clipping', 0.0),
        
        # Parameter optimasi
        ('optimizer_dropdown', optimizer_config, 'type', 'Adam'),
        ('learning_rate_slider', optimizer_config, 'learning_rate', 0.01),
        ('weight_decay_slider', optimizer_config, 'weight_decay', 0.0005),
        ('momentum_slider', optimizer_config, 'momentum', 0.937),
        
        # Parameter penjadwalan
        ('scheduler_dropdown', scheduler_config, 'type', 'cosine'),
        ('warmup_epochs_slider', scheduler_config, 'warmup_epochs', 3),
        
        # Parameter regularisasi
        ('augment_checkbox', regularization_config, 'augment', True),
        ('label_smoothing_slider', regularization_config, 'label_smoothing', 0.0),
        
        # Parameter loss
        ('box_loss_gain_slider', loss_config, 'box_loss_gain', 0.05),
        ('cls_loss_gain_slider', loss_config, 'cls_loss_gain', 0.5),
        ('obj_loss_gain_slider', loss_config, 'obj_loss_gain', 1.0),
        
        # Early stopping parameters
        ('early_stopping_checkbox', early_stopping_config, 'enabled', True),
        ('patience_slider', early_stopping_config, 'patience', 15),
        ('min_delta_slider', early_stopping_config, 'min_delta', 0.001),
        
        # Checkpoint parameters
        ('save_best_checkbox', save_best_config, 'enabled', True),
        ('checkpoint_metric_dropdown', save_best_config, 'metric', 'mAP_0.5')
    ]
    
    # Apply all updates dengan one-liner
    [set_val(component_key, source_config.get(config_key, default_value)) 
     for component_key, source_config, config_key, default_value in field_mappings 
     if component_key in ui_components]
    
    # Logging untuk informasi
    if 'status_panel' in ui_components:
        from smartcash.ui.utils.alert_utils import update_status_panel
        update_status_panel(ui_components['status_panel'], f"📈 UI hyperparameters berhasil diperbarui", "info")