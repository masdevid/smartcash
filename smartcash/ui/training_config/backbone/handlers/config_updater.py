"""
File: smartcash/ui/training_config/backbone/handlers/config_updater.py
Deskripsi: Update UI components dari configuration dengan one-liner assignment
"""

from typing import Dict, Any

def update_backbone_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update UI components dari backbone configuration sesuai struktur model_config.yaml.
    
    Args:
        ui_components: Dictionary UI components
        config: Configuration dictionary
    """
    model_config = config.get('model', {})
    transfer_learning_config = model_config.get('transfer_learning', {})
    unfreeze_schedule = transfer_learning_config.get('unfreeze_schedule', {})
    efficientnet_config = model_config.get('efficientnet', {})
    attention_config = efficientnet_config.get('attention', {})
    export_config = model_config.get('export', {})
    experiments_config = model_config.get('experiments', {})
    
    # Inisialisasi flag untuk mencegah loop event handler
    ui_components['_suppress_backbone_change'] = True
    ui_components['_suppress_optimization_change'] = True
    
    # One-liner safe setter
    safe_update = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Update backbone dan model type dropdowns
    backbone_value = model_config.get('backbone', 'efficientnet_b4')
    model_type = model_config.get('type', 'efficient_optimized')
    
    # Update UI components dengan mapping approach
    field_mappings = [
        # Backbone dan model type
        ('backbone_dropdown', backbone_value),
        ('model_type_dropdown', model_type),
        
        # Backbone settings
        ('freeze_backbone_checkbox', model_config.get('backbone_freeze', False)),
        ('unfreeze_epoch_slider', model_config.get('backbone_unfreeze_epoch', 5)),
        
        # Thresholds
        ('confidence_threshold_slider', model_config.get('confidence', 0.25)),
        ('iou_threshold_slider', model_config.get('iou_threshold', 0.45)),
        
        # Transfer learning
        ('transfer_learning_checkbox', transfer_learning_config.get('enabled', True)),
        ('pretrained_weights_text', transfer_learning_config.get('weights', '')),
        ('freeze_layers_slider', transfer_learning_config.get('freeze_layers', 0)),
        ('unfreeze_schedule_checkbox', unfreeze_schedule.get('enabled', True)),
        ('unfreeze_start_epoch_slider', unfreeze_schedule.get('start_epoch', 5)),
        ('layers_per_epoch_slider', unfreeze_schedule.get('layers_per_epoch', 1)),
        
        # Feature optimization
        ('use_attention_checkbox', model_config.get('use_attention', True)),
        ('use_residual_checkbox', model_config.get('use_residual', True)),
        ('use_ciou_checkbox', model_config.get('use_ciou', False)),
        
        # Optimasi model
        ('quantization_checkbox', model_config.get('quantization', False)),
        ('qat_checkbox', model_config.get('quantization_aware_training', False)),
        ('fp16_checkbox', model_config.get('fp16_training', True)),
        
        # Export settings
        ('quantize_export_checkbox', export_config.get('quantize_export', False)),
        
        # Eksperimen
        ('experiments_enabled_checkbox', experiments_config.get('enabled', False)),
        ('experiment_scenario_dropdown', experiments_config.get('scenario', 'baseline'))
    ]
    
    # Apply all updates
    [safe_update(key, value) for key, value in field_mappings if key in ui_components]
    
    # Update dependent UI berdasarkan backbone selection
    _update_dependent_ui(ui_components, backbone_value)
    
    # Reset flag setelah update UI selesai
    ui_components['_suppress_backbone_change'] = False
    ui_components['_suppress_optimization_change'] = False
    
    # Logging untuk informasi
    if 'status_panel' in ui_components:
        from smartcash.ui.utils.alert_utils import update_status_panel
        update_status_panel(ui_components['status_panel'], f"🤖 UI backbone berhasil diperbarui", "info")

def _update_dependent_ui(ui_components: Dict[str, Any], backbone: str) -> None:
    """Update UI yang dependent pada backbone selection"""
    # One-liner conditional UI updates
    is_cspdarknet = backbone == 'cspdarknet_s'
    
    # Update checkboxes disabled state dan values
    checkbox_configs = [
        ('use_attention_checkbox', is_cspdarknet, False),
        ('use_residual_checkbox', is_cspdarknet, False), 
        ('use_ciou_checkbox', is_cspdarknet, False)
    ]
    
    [setattr(ui_components.get(cb_name), 'disabled', disabled) or 
     setattr(ui_components.get(cb_name), 'value', value if disabled else getattr(ui_components.get(cb_name), 'value', False))
     if hasattr(ui_components.get(cb_name), 'value') else None
     for cb_name, disabled, value in checkbox_configs]