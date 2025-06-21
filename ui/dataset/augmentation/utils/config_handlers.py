"""
File: smartcash/ui/dataset/augmentation/utils/config_handlers.py
Deskripsi: Fixed config handlers dengan proper button event handling
"""

from typing import Dict, Any

def handle_save_config(ui_components: Dict[str, Any]):
    """Handle save config dengan fixed button event handling"""
    def _save_handler(button=None):
        from smartcash.ui.dataset.augmentation.utils.ui_utils import validate_form_inputs, clear_ui_outputs
        
        clear_ui_outputs(ui_components)
        _clear_confirmation_area(ui_components)
        
        # Enhanced form validation
        validation = validate_form_inputs(ui_components)
        if not validation['valid']:
            _show_validation_errors_in_area(ui_components, validation)
            return
        
        # Show enhanced config summary confirmation
        _show_enhanced_config_summary_confirmation(ui_components)
    
    return _save_handler

def handle_reset_config(ui_components: Dict[str, Any]):
    """Handle reset config dengan fixed button event handling"""
    def _reset_handler(button=None):
        from smartcash.ui.dataset.augmentation.utils.ui_utils import clear_ui_outputs
        
        clear_ui_outputs(ui_components)
        _clear_confirmation_area(ui_components)
        
        # Execute reset langsung
        _execute_config_operation(ui_components, 'reset')
    
    return _reset_handler

def _execute_config_operation(ui_components: Dict[str, Any], operation_type: str):
    """Consolidated config operation execution"""
    try:
        config_handler = ui_components.get('config_handler')
        if not config_handler:
            from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
            log_to_ui(ui_components, "❌ Config handler tidak tersedia", "error")
            return
        
        # Set UI components untuk logging
        if hasattr(config_handler, 'set_ui_components'):
            config_handler.set_ui_components(ui_components)
        
        # Execute operation
        if operation_type == 'save':
            success = config_handler.save_config(ui_components)
            if success:
                _show_config_success(ui_components, 'save')
        
        elif operation_type == 'reset':
            success = config_handler.reset_config(ui_components)
            if success:
                _show_config_success(ui_components, 'reset')
            
    except Exception as e:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, f"❌ Error {operation_type} config: {str(e)}", "error")

def _show_enhanced_config_summary_confirmation(ui_components: Dict[str, Any]):
    """Show enhanced config summary dengan HSV dan cleanup target"""
    from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
    
    # Extract current config
    current_config = extract_augmentation_config(ui_components)
    aug_config = current_config.get('augmentation', {})
    cleanup_config = current_config.get('cleanup', {})
    
    # Create enhanced summary
    basic_items = [
        f"🎯 Variations: {aug_config.get('num_variations', 3)}",
        f"📊 Target Count: {aug_config.get('target_count', 500)}",
        f"🔄 Types: {', '.join(aug_config.get('types', ['combined']))}",
        f"📂 Split: {aug_config.get('target_split', 'train')}",
        f"🎚️ Intensity: {aug_config.get('intensity', 0.7)}"
    ]
    
    # HSV parameters
    lighting_config = aug_config.get('lighting', {})
    hsv_items = [
        f"💡 Brightness: {lighting_config.get('brightness_limit', 0.2)}",
        f"🌈 HSV Hue: {lighting_config.get('hsv_hue', 10)}",
        f"🎨 HSV Saturation: {lighting_config.get('hsv_saturation', 15)}"
    ]
    
    # Cleanup target
    cleanup_items = [
        f"🧹 Cleanup Target: {cleanup_config.get('default_target', 'both')}"
    ]
    
    message = """
    Simpan konfigurasi dengan pengaturan berikut?

    <div style="background: #f8f9fa; padding: 8px; border-radius: 4px; margin: 8px 0;">
        <strong>⚙️ Basic Settings:</strong><br>
        {}
    </div>
    <div style="background: #fff3cd; padding: 8px; border-radius: 4px; margin: 8px 0;">
        <strong>🎨 Lighting & HSV:</strong><br>
        {}
    </div>
    <div style="background: #d1ecf1; padding: 8px; border-radius: 4px; margin: 8px 0;">
        <strong>🧹 Cleanup Settings:</strong><br>
        {}
    </div>
    """.format(
        ' • '.join(basic_items),
        ' • '.join(hsv_items),
        ' • '.join(cleanup_items)
    )
    
    def on_confirm_save(btn):
        _clear_confirmation_area(ui_components)
        _execute_config_operation(ui_components, 'save')
    
    def on_cancel_save(btn):
        _clear_confirmation_area(ui_components)
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, "❌ Save config dibatalkan", "info")
    
    _show_confirmation_in_area(
        ui_components,
        title="Konfirmasi Save Konfigurasi",
        message=message,
        on_confirm=on_confirm_save,
        on_cancel=on_cancel_save,
        confirm_text="Ya, Simpan",
        cancel_text="Batal"
    )

def _show_validation_errors_in_area(ui_components: Dict[str, Any], validation: Dict[str, Any]):
    """Show enhanced validation errors"""
    error_messages = validation.get('errors', [])
    warning_messages = validation.get('warnings', [])
    
    # Categorize warnings
    hsv_warnings = [w for w in warning_messages if 'HSV' in w]
    cleanup_warnings = [w for w in warning_messages if 'Cleanup' in w or 'cleanup' in w]
    other_warnings = [w for w in warning_messages if w not in hsv_warnings and w not in cleanup_warnings]
    
    message_parts = []
    
    if error_messages:
        message_parts.append("<strong style='color: #dc3545;'>❌ Errors:</strong>")
        message_parts.extend([f"<div style='margin-left: 15px; color: #dc3545;'>{error}</div>" for error in error_messages])
    
    if hsv_warnings:
        message_parts.append("<strong style='color: #ffc107;'>🎨 HSV Warnings:</strong>")
        message_parts.extend([f"<div style='margin-left: 15px; color: #ffc107;'>{warning}</div>" for warning in hsv_warnings])
    
    if cleanup_warnings:
        message_parts.append("<strong style='color: #17a2b8;'>🧹 Cleanup Warnings:</strong>")
        message_parts.extend([f"<div style='margin-left: 15px; color: #17a2b8;'>{warning}</div>" for warning in cleanup_warnings])
    
    if other_warnings:
        message_parts.append("<strong style='color: #ffc107;'>⚠️ Other Warnings:</strong>")
        message_parts.extend([f"<div style='margin-left: 15px; color: #ffc107;'>{warning}</div>" for warning in other_warnings])
    
    message = "<br>".join(message_parts) + """
    <div style='margin-top: 10px; color: #666;'>
    💡 Silakan perbaiki input form sebelum menyimpan.
    </div>"""
    
    _show_warning_in_area(
        ui_components,
        title="Validation Error - Save Config",
        message=message
    )

def _show_config_success(ui_components: Dict[str, Any], operation_type: str):
    """Show config operation success"""
    if operation_type == 'save':
        title = "Save Config Berhasil"
        message = "✅ Konfigurasi berhasil disimpan!"
    elif operation_type == 'reset':
        title = "Reset Config Berhasil" 
        message = "✅ Konfigurasi berhasil direset!"
    
    _show_info_in_area(ui_components, title=title, message=message)

# Direct confirmation area utilities (replace dialog_utils dependency)
def _clear_confirmation_area(ui_components: Dict[str, Any]):
    """Clear confirmation area directly"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'clear_output'):
        confirmation_area.clear_output(wait=True)

def _show_confirmation_in_area(ui_components: Dict[str, Any], title: str, message: str, 
                              on_confirm=None, on_cancel=None, confirm_text="OK", cancel_text="Cancel"):
    """Show confirmation dialog in area"""
    from IPython.display import display, HTML
    import ipywidgets as widgets
    
    confirmation_area = ui_components.get('confirmation_area')
    if not confirmation_area:
        return
    
    # Create buttons
    confirm_btn = widgets.Button(description=confirm_text, button_style='primary')
    cancel_btn = widgets.Button(description=cancel_text, button_style='')
    
    if on_confirm:
        confirm_btn.on_click(on_confirm)
    if on_cancel:
        cancel_btn.on_click(on_cancel)
    
    # Create dialog content
    dialog_html = f"""
    <div style="background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0;">
        <h5 style="margin-top: 0; color: #333;">{title}</h5>
        {message}
    </div>
    """
    
    button_container = widgets.HBox([confirm_btn, cancel_btn], 
                                   layout=widgets.Layout(justify_content='center', margin='10px 0'))
    
    with confirmation_area:
        display(HTML(dialog_html))
        display(button_container)

def _show_warning_in_area(ui_components: Dict[str, Any], title: str, message: str):
    """Show warning in confirmation area"""
    from IPython.display import display, HTML
    
    confirmation_area = ui_components.get('confirmation_area')
    if not confirmation_area:
        return
    
    warning_html = f"""
    <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 15px; margin: 10px 0;">
        <h5 style="margin-top: 0; color: #856404;">{title}</h5>
        {message}
    </div>
    """
    
    with confirmation_area:
        display(HTML(warning_html))

def _show_info_in_area(ui_components: Dict[str, Any], title: str, message: str):
    """Show info in confirmation area"""
    from IPython.display import display, HTML
    
    confirmation_area = ui_components.get('confirmation_area')
    if not confirmation_area:
        return
    
    info_html = f"""
    <div style="background: #d1ecf1; border: 1px solid #17a2b8; border-radius: 8px; padding: 15px; margin: 10px 0;">
        <h5 style="margin-top: 0; color: #0c5460;">{title}</h5>
        {message}
    </div>
    """
    
    with confirmation_area:
        display(HTML(info_html))