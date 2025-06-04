"""
File: smartcash/ui/training_config/backbone/handlers/config_updater.py
Deskripsi: Update UI components dari configuration dengan one-liner assignment
"""

from typing import Dict, Any

def update_backbone_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update UI components dari backbone configuration.
    
    Args:
        ui_components: Dictionary UI components
        config: Configuration dictionary
    """
    model_config = config.get('model', {})
    
    # Inisialisasi flag untuk mencegah loop event handler
    ui_components['_suppress_backbone_change'] = True
    ui_components['_suppress_optimization_change'] = True
    
    # Update backbone dropdown
    backbone_value = model_config.get('backbone', 'efficientnet_b4')
    [setattr(ui_components.get('backbone_dropdown'), 'value', backbone_value) 
     if hasattr(ui_components.get('backbone_dropdown'), 'value') else None][0]
    
    # Update model type dropdown - default ke efficient_optimized
    model_type = model_config.get('model_type', 'efficient_optimized')
    [setattr(ui_components.get('model_type_dropdown'), 'value', model_type) 
     if hasattr(ui_components.get('model_type_dropdown'), 'value') else None][0]
    
    # Update optimization checkboxes berdasarkan model_type yang tersimpan
    use_attention = model_config.get('use_attention', True)
    use_residual = model_config.get('use_residual', False)
    use_ciou = model_config.get('use_ciou', False)
    
    [setattr(ui_components.get('use_attention_checkbox'), 'value', use_attention) 
     if hasattr(ui_components.get('use_attention_checkbox'), 'value') else None][0]
    
    [setattr(ui_components.get('use_residual_checkbox'), 'value', use_residual) 
     if hasattr(ui_components.get('use_residual_checkbox'), 'value') else None][0]
    
    [setattr(ui_components.get('use_ciou_checkbox'), 'value', use_ciou) 
     if hasattr(ui_components.get('use_ciou_checkbox'), 'value') else None][0]
    
    # Update dependent UI berdasarkan backbone selection
    _update_dependent_ui(ui_components, backbone_value)
    
    # Reset flag setelah update UI selesai
    ui_components['_suppress_backbone_change'] = False
    ui_components['_suppress_optimization_change'] = False

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