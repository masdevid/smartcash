"""
File: smartcash/ui/training_config/backbone/handlers/form_handlers.py
Deskripsi: Form event handlers untuk backbone configuration dengan one-liner handlers
"""

from typing import Dict, Any

def setup_backbone_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup event handlers untuk backbone form components"""
    # Inisialisasi flag untuk mencegah loop
    ui_components['_suppress_backbone_change'] = False
    ui_components['_suppress_optimization_change'] = False
    
    # Definisikan handler untuk setiap widget
    handlers = [
        ('backbone_dropdown', 'value', lambda change: _on_backbone_change(change, ui_components)),
        ('model_type_dropdown', 'value', lambda change: _on_model_type_change(change, ui_components)),
        ('use_attention_checkbox', 'value', lambda change: _on_optimization_change(change, ui_components)),
        ('use_residual_checkbox', 'value', lambda change: _on_optimization_change(change, ui_components)),
        ('use_ciou_checkbox', 'value', lambda change: _on_optimization_change(change, ui_components))
    ]
    
    # Register handlers dengan one-liner
    [getattr(ui_components.get(widget_name), 'observe', lambda: None)(handler, names=observe_name)
     for widget_name, observe_name, handler in handlers if widget_name in ui_components]

def _on_backbone_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """Handle backbone dropdown change"""
    # Cek apakah update ini harus dilewati (untuk mencegah loop)
    if ui_components.get('_suppress_backbone_change', False):
        return
        
    backbone = change.get('new')
    if not backbone: return
    
    # Konfigurasi untuk setiap backbone
    # Format: (model_type, disable_options, attention, residual, ciou)
    updates = {
        'cspdarknet_s': ('yolov5s', True, False, False, False),
        'efficientnet_b4': ('efficient_optimized', False, True, True, True)  # Default ke optimized
    }
    
    model_type, disable_opts, attention_val, residual_val, ciou_val = updates.get(backbone, ('efficient_optimized', False, True, True, True))
    
    # Update model type dropdown dengan flag untuk mencegah loop
    ui_components['_suppress_optimization_change'] = True
    [setattr(ui_components.get('model_type_dropdown'), 'value', model_type) 
     if hasattr(ui_components.get('model_type_dropdown'), 'value') else None][0]
    
    # Update optimization checkboxes
    _update_optimization_checkboxes(ui_components, disable_opts, attention_val, residual_val, ciou_val)
    ui_components['_suppress_optimization_change'] = False
    
    from smartcash.ui.utils.alert_utils import update_status_panel
    update_status_panel(ui_components.get('status_panel'), f"🔄 Backbone diubah ke {backbone}", "info")

def _on_model_type_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """Handle model type dropdown change"""
    model_type = change.get('new')
    if not model_type: return
    
    # Model type to backbone and optimization settings mapping
    # Format: (backbone, disable_options, attention, residual, ciou)
    model_type_settings = {
        'yolov5s': ('cspdarknet_s', True, False, False, False),
        'efficient_basic': ('efficientnet_b4', False, False, False, False),
        'efficient_optimized': ('efficientnet_b4', False, True, False, False),
        'efficient_advanced': ('efficientnet_b4', False, True, True, True)
    }
    
    backbone, disable_opts, attention_val, residual_val, ciou_val = model_type_settings.get(
        model_type, ('efficientnet_b4', False, True, False, False))
    
    # Update backbone tanpa trigger event handler (mencegah loop)
    ui_components['_suppress_backbone_change'] = True
    [setattr(ui_components.get('backbone_dropdown'), 'value', backbone)
     if hasattr(ui_components.get('backbone_dropdown'), 'value') else None][0]
    ui_components['_suppress_backbone_change'] = False
    
    # Update optimization checkboxes sesuai dengan model type yang dipilih
    _update_optimization_checkboxes(ui_components, disable_opts, attention_val, residual_val, ciou_val)
    
    from smartcash.ui.utils.alert_utils import update_status_panel
    update_status_panel(ui_components.get('status_panel'), f"🔄 Model type diubah ke {model_type}", "info")

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

def _on_optimization_change(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """Handle optimization checkbox changes dan update model_type sesuai"""
    # Cek apakah update ini harus dilewati (untuk mencegah loop)
    if ui_components.get('_suppress_optimization_change', False):
        return
    
    # Backbone harus efficientnet_b4 untuk enabling optimization options
    if ui_components.get('backbone_dropdown').value != 'efficientnet_b4':
        return
    
    # Dapatkan nilai checkbox
    use_attention = ui_components.get('use_attention_checkbox').value
    use_residual = ui_components.get('use_residual_checkbox').value
    use_ciou = ui_components.get('use_ciou_checkbox').value
    
    # Tentukan model type berdasarkan kombinasi checkbox
    if use_attention and use_residual and use_ciou:
        model_type = 'efficient_advanced'
    elif use_attention and not use_residual and not use_ciou:
        model_type = 'efficient_optimized'
    elif not use_attention and not use_residual and not use_ciou:
        model_type = 'efficient_basic'
    else:
        # Kasus lainnya (kombinasi yang tidak standard)
        if use_attention:
            model_type = 'efficient_optimized'  # fallback ke optimized jika ada attention
        else:
            model_type = 'efficient_basic'  # fallback ke basic lainnya
    
    # Set model type tanpa memicu handler untuk menghindari loop
    ui_components['_suppress_backbone_change'] = True  # Mencegah loop
    if ui_components.get('model_type_dropdown').value != model_type:
        ui_components.get('model_type_dropdown').value = model_type
    ui_components['_suppress_backbone_change'] = False
    
    # Update info
    from smartcash.ui.utils.alert_utils import update_status_panel
    update_status_panel(ui_components.get('status_panel'), "⚡ Optimasi diperbarui ke " + model_type, "info")

def _update_optimization_info(ui_components: Dict[str, Any]) -> None:
    """Update info setelah optimization changes (deprecated - retained for compatibility)"""
    from smartcash.ui.utils.alert_utils import update_status_panel
    update_status_panel(ui_components.get('status_panel'), "⚡ Konfigurasi optimasi diperbarui", "info")