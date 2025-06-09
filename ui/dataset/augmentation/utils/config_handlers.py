"""
File: smartcash/ui/dataset/augmentation/utils/config_handlers.py
Deskripsi: Optimized config handlers dengan simplified logic dan consolidated operations
"""

from typing import Dict, Any

def handle_save_config(ui_components: Dict[str, Any]):
    """Handle save config dengan optimized validation dan confirmation"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import validate_form_inputs, clear_ui_outputs
    from smartcash.ui.dataset.augmentation.utils.dialog_utils import clear_confirmation_area
    
    clear_ui_outputs(ui_components)
    clear_confirmation_area(ui_components)
    
    # Form validation
    validation = validate_form_inputs(ui_components)
    if not validation['valid']:
        _show_validation_errors_in_area(ui_components, validation)
        return
    
    # Show config summary confirmation
    _show_config_summary_confirmation(ui_components)

def handle_reset_config(ui_components: Dict[str, Any]):
    """Handle reset config langsung tanpa confirmation"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import clear_ui_outputs
    from smartcash.ui.dataset.augmentation.utils.dialog_utils import clear_confirmation_area
    
    clear_ui_outputs(ui_components)
    clear_confirmation_area(ui_components)
    
    # Execute reset langsung
    _execute_config_operation(ui_components, 'reset')

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

def _show_config_summary_confirmation(ui_components: Dict[str, Any]):
    """Show config summary dengan save confirmation"""
    from smartcash.ui.dataset.augmentation.utils.dialog_utils import show_confirmation_in_area
    from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
    
    # Extract current config
    current_config = extract_augmentation_config(ui_components)
    aug_config = current_config.get('augmentation', {})
    
    # Create summary
    summary_items = [
        f"🎯 Variations: {aug_config.get('num_variations', 3)}",
        f"📊 Target Count: {aug_config.get('target_count', 500)}",
        f"🔄 Types: {', '.join(aug_config.get('types', ['combined']))}",
        f"📂 Split: {aug_config.get('target_split', 'train')}",
        f"⚖️ Balance Classes: {'Yes' if aug_config.get('balance_classes', True) else 'No'}",
        f"🎚️ Intensity: {aug_config.get('intensity', 0.7)}"
    ]
    
    message = "Simpan konfigurasi dengan pengaturan berikut?\n\n" + "\n".join(summary_items)
    
    def on_confirm_save(btn):
        from smartcash.ui.dataset.augmentation.utils.dialog_utils import clear_confirmation_area
        clear_confirmation_area(ui_components)
        _execute_config_operation(ui_components, 'save')
    
    def on_cancel_save(btn):
        from smartcash.ui.dataset.augmentation.utils.dialog_utils import clear_confirmation_area
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        clear_confirmation_area(ui_components)
        log_to_ui(ui_components, "❌ Save config dibatalkan", "info")
    
    show_confirmation_in_area(
        ui_components,
        title="Konfirmasi Save Konfigurasi",
        message=message,
        on_confirm=on_confirm_save,
        on_cancel=on_cancel_save,
        confirm_text="Ya, Simpan",
        cancel_text="Batal"
    )

def _show_validation_errors_in_area(ui_components: Dict[str, Any], validation: Dict[str, Any]):
    """Show validation errors di confirmation area"""
    from smartcash.ui.dataset.augmentation.utils.dialog_utils import show_warning_in_area
    
    error_messages = validation.get('errors', []) + validation.get('warnings', [])
    message = "Form validation menemukan masalah:\n\n" + "\n".join(error_messages)
    message += "\n\nSilakan perbaiki input form sebelum menyimpan."
    
    show_warning_in_area(
        ui_components,
        title="Validation Error - Save Config",
        message=message,
        on_close=lambda btn: None
    )

def _show_config_success(ui_components: Dict[str, Any], operation_type: str):
    """Show config operation success di confirmation area"""
    from smartcash.ui.dataset.augmentation.utils.dialog_utils import show_info_in_area
    
    if operation_type == 'save':
        title = "Save Config Berhasil"
        message = "✅ Konfigurasi berhasil disimpan!\n\nKonfigurasi telah disimpan ke file dan UI telah direfresh dengan nilai terbaru."
    elif operation_type == 'reset':
        title = "Reset Config Berhasil"  
        message = "✅ Konfigurasi berhasil direset!\n\nSemua pengaturan telah dikembalikan ke nilai default dan UI telah diperbarui."
    else:
        title = "Config Operation Berhasil"
        message = f"✅ {operation_type.title()} config berhasil!"
    
    show_info_in_area(
        ui_components,
        title=title,
        message=message,
        on_close=lambda btn: None
    )

# Utility functions untuk compatibility
def validate_and_save_config(ui_components: Dict[str, Any]) -> bool:
    """Validate dan save config - return success status"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import validate_form_inputs
    
    validation = validate_form_inputs(ui_components)
    if not validation['valid']:
        return False
    
    try:
        config_handler = ui_components.get('config_handler')
        if not config_handler:
            return False
        
        if hasattr(config_handler, 'set_ui_components'):
            config_handler.set_ui_components(ui_components)
        
        return config_handler.save_config(ui_components)
    except Exception:
        return False

def reset_config_silent(ui_components: Dict[str, Any]) -> bool:
    """Reset config tanpa UI feedback - return success status"""
    try:
        config_handler = ui_components.get('config_handler')
        if not config_handler:
            return False
        
        if hasattr(config_handler, 'set_ui_components'):
            config_handler.set_ui_components(ui_components)
        
        return config_handler.reset_config(ui_components)
    except Exception:
        return False

def get_current_config_summary(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get current config summary untuk display"""
    try:
        from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
        
        config = extract_augmentation_config(ui_components)
        aug_config = config.get('augmentation', {})
        
        return {
            'num_variations': aug_config.get('num_variations', 2),
            'target_count': aug_config.get('target_count', 500),
            'intensity': aug_config.get('intensity', 0.7),
            'types': aug_config.get('types', ['combined']),
            'target_split': aug_config.get('target_split', 'train'),
            'balance_classes': aug_config.get('balance_classes', True),
            'normalization_method': config.get('preprocessing', {}).get('normalization', {}).get('method', 'minmax')
        }
    except Exception:
        return {}

# One-liner utilities
safe_save = lambda ui_components: validate_and_save_config(ui_components)
safe_reset = lambda ui_components: reset_config_silent(ui_components)
get_summary = lambda ui_components: get_current_config_summary(ui_components)