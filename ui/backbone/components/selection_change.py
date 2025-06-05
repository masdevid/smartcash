"""
File: smartcash/ui/backbone/components/selection_change.py
Deskripsi: Fixed handlers untuk perubahan selection dengan status panel yang tepat dan checkbox sync yang benar
"""

from typing import Dict, Any

def setup_backbone_selection_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup handlers dengan event suppression dan debounced status updates"""
    from ..handlers.defaults import get_model_type_mapping, get_backbone_mapping
    
    # Global suppression flag untuk prevent cascading events
    ui_components['_suppress_all_changes'] = False
    ui_components['_status_timer'] = None  # For debounced status updates
    
    # Mapping data untuk handlers
    model_type_mapping = get_model_type_mapping()
    backbone_mapping = get_backbone_mapping()
    
    # Handler definitions dengan global suppression
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
    """Handle backbone change dengan global suppression"""
    if ui_components.get('_suppress_all_changes', False):
        return
    
    backbone = change.get('new')
    if not backbone or backbone not in backbone_mapping:
        return
    
    # Block all other change events
    ui_components['_suppress_all_changes'] = True
    
    try:
        mapping = backbone_mapping[backbone]
        
        # Update model_type dropdown
        if 'model_type_dropdown' in ui_components:
            ui_components['model_type_dropdown'].value = mapping['default_model_type']
        
        # Force update checkboxes
        _force_update_feature_checkboxes(ui_components, mapping)
        
        # Debounced status update untuk avoid overlap dengan save/reset messages
        _debounced_status_update(ui_components, f"Backbone {backbone} dipilih", "info")
        
    finally:
        ui_components['_suppress_all_changes'] = False

def _on_model_type_change(change: Dict[str, Any], ui_components: Dict[str, Any], model_type_mapping: Dict[str, Any]) -> None:
    """Handle model_type change dengan global suppression"""
    if ui_components.get('_suppress_all_changes', False):
        return
    
    model_type = change.get('new')
    if not model_type or model_type not in model_type_mapping:
        return
    
    # Block all other change events
    ui_components['_suppress_all_changes'] = True
    
    try:
        mapping = model_type_mapping[model_type]
        
        # Update backbone dropdown
        if 'backbone_dropdown' in ui_components:
            ui_components['backbone_dropdown'].value = mapping['backbone']
        
        # Force reset dan update checkboxes
        _force_update_feature_checkboxes(ui_components, mapping)
        
        # Debounced status update
        _debounced_status_update(ui_components, f"Model {model_type} dipilih", "info")
        
    finally:
        ui_components['_suppress_all_changes'] = False

def _on_feature_change(ui_components: Dict[str, Any]) -> None:
    """Handle feature changes dengan global suppression"""
    if ui_components.get('_suppress_all_changes', False):
        return
    
    # Block all other change events
    ui_components['_suppress_all_changes'] = True
    
    try:
        # Get current values
        backbone = getattr(ui_components.get('backbone_dropdown'), 'value', 'efficientnet_b4')
        use_attention = getattr(ui_components.get('use_attention_checkbox'), 'value', False)
        use_residual = getattr(ui_components.get('use_residual_checkbox'), 'value', False)
        use_ciou = getattr(ui_components.get('use_ciou_checkbox'), 'value', False)
        
        # Auto-detect model_type
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
            if 'model_type_dropdown' in ui_components:
                ui_components['model_type_dropdown'].value = detected_model_type
            _debounced_status_update(ui_components, f"Auto-detect: {detected_model_type}", "success")
        
    finally:
        ui_components['_suppress_all_changes'] = False

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

def _debounced_status_update(ui_components: Dict[str, Any], message: str, status_type: str = "info", delay_ms: int = 500) -> None:
    """Debounced status update untuk avoid overlap dengan save/reset messages"""
    import threading
    
    # Cancel previous timer jika ada
    if ui_components.get('_status_timer'):
        ui_components['_status_timer'].cancel()
    
    # Cek apakah status panel menampilkan save/reset message
    current_status = getattr(ui_components.get('status_panel'), 'value', '')
    if any(keyword in current_status for keyword in ['tersimpan', 'direset', 'Menyimpan', 'Mereset']):
        return  # Don't override save/reset messages
    
    # Set timer untuk delayed update
    def delayed_update():
        _update_status_panel(ui_components, message, status_type)
    
    timer = threading.Timer(delay_ms / 1000.0, delayed_update)
    ui_components['_status_timer'] = timer
    timer.start()

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """Update status panel dengan message"""
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components:
        update_status_panel(ui_components['status_panel'], message, status_type)