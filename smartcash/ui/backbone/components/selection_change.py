"""
File: smartcash/ui/backbone/components/selection_change.py
Deskripsi: Fixed handlers untuk perubahan selection dengan status panel yang tepat dan checkbox sync yang benar
"""

from typing import Dict, Any

def setup_backbone_selection_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup handlers untuk selection changes dengan improved sinkronisasi"""
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
    """Handle backbone dropdown change dengan fixed status update"""
    if ui_components.get('_suppress_backbone_change', False):
        return
    
    backbone = change.get('new')
    if not backbone or backbone not in backbone_mapping:
        return
    
    # Get mapping untuk backbone yang dipilih
    mapping = backbone_mapping[backbone]
    
    # Suppress model_type change untuk prevent loop
    ui_components['_suppress_model_type_change'] = True
    
    try:
        # Update model_type dropdown
        if 'model_type_dropdown' in ui_components:
            ui_components['model_type_dropdown'].value = mapping['default_model_type']
        
        # Update feature checkboxes
        _update_feature_checkboxes(ui_components, mapping)
        
        # Fixed status panel update dengan single icon
        _update_status_panel(ui_components, f"Backbone {backbone} dipilih", "info")
        
    finally:
        ui_components['_suppress_model_type_change'] = False

def _on_model_type_change(change: Dict[str, Any], ui_components: Dict[str, Any], model_type_mapping: Dict[str, Any]) -> None:
    """Handle model_type dropdown change dengan fixed checkbox sync"""
    if ui_components.get('_suppress_model_type_change', False):
        return
    
    model_type = change.get('new')
    if not model_type or model_type not in model_type_mapping:
        return
    
    # Get mapping untuk model_type yang dipilih
    mapping = model_type_mapping[model_type]
    
    # Suppress backbone change untuk prevent loop
    ui_components['_suppress_backbone_change'] = True
    
    try:
        # Update backbone dropdown
        if 'backbone_dropdown' in ui_components:
            ui_components['backbone_dropdown'].value = mapping['backbone']
        
        # Fixed checkbox update - force nilai dari mapping
        _force_update_feature_checkboxes(ui_components, mapping)
        
        # Fixed status panel update dengan single icon
        _update_status_panel(ui_components, f"Model {model_type} dipilih", "info")
        
    finally:
        ui_components['_suppress_backbone_change'] = False

def _on_feature_change(ui_components: Dict[str, Any]) -> None:
    """Handle feature checkbox changes dengan improved auto-detect"""
    if ui_components.get('_suppress_model_type_change', False):
        return
    
    # Get current checkbox values dengan safe access
    backbone = getattr(ui_components.get('backbone_dropdown'), 'value', 'efficientnet_b4')
    use_attention = getattr(ui_components.get('use_attention_checkbox'), 'value', False)
    use_residual = getattr(ui_components.get('use_residual_checkbox'), 'value', False)
    use_ciou = getattr(ui_components.get('use_ciou_checkbox'), 'value', False)
    
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
        detected_model_type = 'efficient_optimized' if use_attention else 'efficient_basic'
    
    # Update model_type jika berbeda
    current_model_type = getattr(ui_components.get('model_type_dropdown'), 'value', '')
    if current_model_type != detected_model_type:
        ui_components['_suppress_backbone_change'] = True
        try:
            if 'model_type_dropdown' in ui_components:
                ui_components['model_type_dropdown'].value = detected_model_type
            _update_status_panel(ui_components, f"Auto-detect: {detected_model_type}", "success")
        finally:
            ui_components['_suppress_backbone_change'] = False

def _update_feature_checkboxes(ui_components: Dict[str, Any], mapping: Dict[str, Any]) -> None:
    """Update feature checkboxes berdasarkan mapping dengan soft update"""
    checkbox_updates = [
        ('use_attention_checkbox', mapping.get('disable_features', False), mapping.get('use_attention', False)),
        ('use_residual_checkbox', mapping.get('disable_features', False), mapping.get('use_residual', False)),
        ('use_ciou_checkbox', mapping.get('disable_features', False), mapping.get('use_ciou', False))
    ]
    
    # Apply updates dengan one-liner - hanya disable jika perlu
    [setattr(ui_components.get(widget_name), 'disabled', disabled) if widget_name in ui_components else None
     for widget_name, disabled, _ in checkbox_updates]

def _force_update_feature_checkboxes(ui_components: Dict[str, Any], mapping: Dict[str, Any]) -> None:
    """Force reset dan update feature checkboxes dengan nilai dari mapping"""
    checkbox_updates = [
        ('use_attention_checkbox', mapping.get('disable_features', False), mapping.get('use_attention', False)),
        ('use_residual_checkbox', mapping.get('disable_features', False), mapping.get('use_residual', False)),
        ('use_ciou_checkbox', mapping.get('disable_features', False), mapping.get('use_ciou', False))
    ]
    
    # Force reset value first, then update disabled state - fix untuk YOLOv5s
    for widget_name, disabled, value in checkbox_updates:
        if widget_name in ui_components:
            widget = ui_components[widget_name]
            widget.value = value  # Reset value first
            widget.disabled = disabled  # Then set disabled state

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """Update status panel dengan single icon dan clean message"""
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components:
        update_status_panel(ui_components['status_panel'], message, status_type)