"""
File: smartcash/ui/backbone/components/selection_change.py
Deskripsi: Handlers untuk perubahan selection backbone dan model_type dengan sinkronisasi otomatis
"""

from typing import Dict, Any

def setup_backbone_selection_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup handlers untuk selection changes dengan sinkronisasi backbone <-> model_type"""
    from ..handlers.defaults import get_model_type_mapping, get_backbone_mapping
    
    # Flags untuk mencegah recursive updates
    ui_components['_suppress_backbone_change'] = False
    ui_components['_suppress_model_type_change'] = False
    
    # Mapping data untuk handlers
    model_type_mapping = get_model_type_mapping()
    backbone_mapping = get_backbone_mapping()
    
    # Handler definitions dengan one-liner registrations
    handlers = [
        ('backbone_dropdown', 'value', lambda change: _on_backbone_change(change, ui_components, backbone_mapping)),
        ('model_type_dropdown', 'value', lambda change: _on_model_type_change(change, ui_components, model_type_mapping)),
        ('use_attention_checkbox', 'value', lambda change: _on_feature_change(ui_components)),
        ('use_residual_checkbox', 'value', lambda change: _on_feature_change(ui_components)),
        ('use_ciou_checkbox', 'value', lambda change: _on_feature_change(ui_components))
    ]
    
    # Register all handlers dengan one-liner
    [getattr(ui_components.get(widget), 'observe', lambda: None)(handler, names=observe_name)
     for widget, observe_name, handler in handlers if widget in ui_components]

def _on_backbone_change(change: Dict[str, Any], ui_components: Dict[str, Any], backbone_mapping: Dict[str, Any]) -> None:
    """Handle backbone dropdown change dan update model_type + features"""
    if ui_components.get('_suppress_backbone_change', False):
        return
    
    backbone = change.get('new')
    if not backbone or backbone not in backbone_mapping:
        return
    
    # Get mapping untuk backbone yang dipilih
    mapping = backbone_mapping[backbone]
    
    # Suppress model_type change untuk prevent loop
    ui_components['_suppress_model_type_change'] = True
    
    # Update model_type dropdown
    safe_update = lambda widget, value: setattr(widget, 'value', value) if hasattr(widget, 'value') else None
    safe_update(ui_components.get('model_type_dropdown'), mapping['default_model_type'])
    
    # Update feature checkboxes
    _update_feature_checkboxes(ui_components, mapping)
    
    # Update status panel
    _update_status_panel(ui_components, f"🔄 Backbone {backbone} dipilih", "info")
    
    ui_components['_suppress_model_type_change'] = False

def _on_model_type_change(change: Dict[str, Any], ui_components: Dict[str, Any], model_type_mapping: Dict[str, Any]) -> None:
    """Handle model_type dropdown change dan update backbone + features"""
    if ui_components.get('_suppress_model_type_change', False):
        return
    
    model_type = change.get('new')
    if not model_type or model_type not in model_type_mapping:
        return
    
    # Get mapping untuk model_type yang dipilih
    mapping = model_type_mapping[model_type]
    
    # Suppress backbone change untuk prevent loop
    ui_components['_suppress_backbone_change'] = True
    
    # Update backbone dropdown
    safe_update = lambda widget, value: setattr(widget, 'value', value) if hasattr(widget, 'value') else None
    safe_update(ui_components.get('backbone_dropdown'), mapping['backbone'])
    
    # Update feature checkboxes
    _update_feature_checkboxes(ui_components, mapping)
    
    # Update status panel
    _update_status_panel(ui_components, f"⚙️ Model type {model_type} dipilih", "info")
    
    ui_components['_suppress_backbone_change'] = False

def _on_feature_change(ui_components: Dict[str, Any]) -> None:
    """Handle feature checkbox changes dan auto-detect model_type yang sesuai"""
    if ui_components.get('_suppress_model_type_change', False):
        return
    
    # Get current checkbox values
    backbone = ui_components.get('backbone_dropdown').value
    use_attention = ui_components.get('use_attention_checkbox').value
    use_residual = ui_components.get('use_residual_checkbox').value
    use_ciou = ui_components.get('use_ciou_checkbox').value
    
    # Auto-detect model_type berdasarkan feature combination
    if backbone == 'cspdarknet_s':
        detected_model_type = 'yolov5s'
    elif use_attention and use_residual and use_ciou:
        detected_model_type = 'efficient_advanced'
    elif use_attention and not use_residual and not use_ciou:
        detected_model_type = 'efficient_optimized'
    elif not use_attention and not use_residual and not use_ciou:
        detected_model_type = 'efficient_basic'
    else:
        # Partial features - default ke optimized jika ada attention
        detected_model_type = 'efficient_optimized' if use_attention else 'efficient_basic'
    
    # Update model_type jika berbeda
    current_model_type = ui_components.get('model_type_dropdown').value
    if current_model_type != detected_model_type:
        ui_components['_suppress_backbone_change'] = True
        ui_components.get('model_type_dropdown').value = detected_model_type
        _update_status_panel(ui_components, f"🔧 Auto-detect: {detected_model_type}", "success")
        ui_components['_suppress_backbone_change'] = False

def _update_feature_checkboxes(ui_components: Dict[str, Any], mapping: Dict[str, Any]) -> None:
    """Update feature checkboxes berdasarkan mapping dengan one-liner updates"""
    checkbox_updates = [
        ('use_attention_checkbox', mapping.get('disable_features', False), mapping.get('use_attention', False)),
        ('use_residual_checkbox', mapping.get('disable_features', False), mapping.get('use_residual', False)),
        ('use_ciou_checkbox', mapping.get('disable_features', False), mapping.get('use_ciou', False))
    ]
    
    # Apply updates dengan one-liner: (widget, disabled, value)
    [setattr(ui_components.get(widget_name), 'disabled', disabled) or 
     setattr(ui_components.get(widget_name), 'value', value if disabled else ui_components.get(widget_name).value)
     for widget_name, disabled, value in checkbox_updates if widget_name in ui_components]

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """Update status panel dengan message"""
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components:
        update_status_panel(ui_components['status_panel'], message, status_type)