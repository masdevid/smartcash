"""
File: smartcash/ui/training_config/backbone/handlers/form_handlers.py
Deskripsi: Form event handlers untuk backbone configuration dengan one-liner handlers
"""

from typing import Dict, Any

def setup_backbone_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup event handlers untuk backbone form components"""
    # One-liner handler registration
    handlers = [
        ('backbone_dropdown', 'value', lambda change: _on_backbone_change(change, ui_components)),
        ('model_type_dropdown', 'value', lambda change: _on_model_type_change(change, ui_components)),
        ('use_attention_checkbox', 'value', lambda change: _update_optimization_info(ui_components)),
        ('use_residual_checkbox', 'value', lambda change: _update_optimization_info(ui_components)),
        ('use_ciou_checkbox', 'value', lambda change: _update_optimization_info(ui_components))
    ]
    
    # Register handlers dengan one-liner
    [getattr(ui_components.get(widget_name), 'observe', lambda: None)(handler, names=observe_name)
     for widget_name, observe_name, handler in handlers if widget_name in ui_components]

def _on_backbone_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """Handle backbone dropdown change"""
    backbone = change.get('new')
    if not backbone: return
    
    # One-liner dependent updates
    updates = {
        'cspdarknet_s': ('yolov5s', True, False, False, False),
        'efficientnet_b4': ('efficient_basic', False, True, True, False)
    }
    
    model_type, disable_opts, attention_val, residual_val, ciou_val = updates.get(backbone, ('efficient_basic', False, True, True, False))
    
    # Update model type dropdown
    [setattr(ui_components.get('model_type_dropdown'), 'value', model_type) 
     if hasattr(ui_components.get('model_type_dropdown'), 'value') else None][0]
    
    # Update optimization checkboxes
    _update_optimization_checkboxes(ui_components, disable_opts, attention_val, residual_val, ciou_val)

def _on_model_type_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """Handle model type dropdown change"""
    model_type = change.get('new')
    if not model_type: return
    
    # One-liner model type to backbone mapping
    backbone_map = {'yolov5s': 'cspdarknet_s', 'efficient_basic': 'efficientnet_b4'}
    backbone = backbone_map.get(model_type, 'efficientnet_b4')
    
    [setattr(ui_components.get('backbone_dropdown'), 'value', backbone)
     if hasattr(ui_components.get('backbone_dropdown'), 'value') else None][0]

def _update_optimization_checkboxes(ui_components: Dict[str, Any], disable: bool, attention: bool, residual: bool, ciou: bool) -> None:
    """Update optimization checkboxes dengan one-liner pattern"""
    checkbox_updates = [
        ('use_attention_checkbox', disable, attention),
        ('use_residual_checkbox', disable, residual), 
        ('use_ciou_checkbox', disable, ciou)
    ]
    
    [setattr(ui_components.get(cb_name), 'disabled', disabled) or 
     (setattr(ui_components.get(cb_name), 'value', value) if disabled else None)
     if hasattr(ui_components.get(cb_name), 'value') else None
     for cb_name, disabled, value in checkbox_updates]

def _update_optimization_info(ui_components: Dict[str, Any]) -> None:
    """Update info setelah optimization changes"""
    from smartcash.ui.utils.alert_utils import update_status_panel
    update_status_panel(ui_components.get('status_panel'), "⚡ Konfigurasi optimasi diperbarui", "info")