"""
File: smartcash/ui/dataset/augmentation/utils/config_handlers.py
Deskripsi: Config handlers dengan dialog integration dan comprehensive validation
"""

from typing import Dict, Any

def handle_save_config(ui_components: Dict[str, Any]):
    """Handle save config dengan validation dan confirmation"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import validate_form_inputs, clear_ui_outputs
    from smartcash.ui.dataset.augmentation.utils.button_manager import with_button_management
    
    @with_button_management
    def _save_operation(ui_components):
        clear_ui_outputs(ui_components)
        
        # Form validation
        validation = validate_form_inputs(ui_components)
        if not validation['valid']:
            _show_save_validation_errors(ui_components, validation)
            return
        
        # Show config summary before save
        _show_config_summary_confirmation(ui_components)
    
    _save_operation(ui_components)

def handle_reset_config(ui_components: Dict[str, Any]):
    """Handle reset config dengan confirmation dialog"""
    from smartcash.ui.components.dialogs import show_confirmation
    from smartcash.ui.dataset.augmentation.utils.ui_utils import clear_ui_outputs
    
    def on_confirm_reset(button):
        clear_ui_outputs(ui_components)
        _execute_reset_config(ui_components)
    
    def on_cancel_reset(button):
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, "❌ Reset dibatalkan", "info")
    
    # Show confirmation dialog
    show_confirmation(
        title="Konfirmasi Reset Konfigurasi",
        message="Apakah Anda yakin ingin reset semua pengaturan ke nilai default?\n\nSemua konfigurasi saat ini akan hilang.",
        on_confirm=on_confirm_reset,
        on_cancel=on_cancel_reset
    )

def _show_save_validation_errors(ui_components: Dict[str, Any], validation: Dict[str, Any]):
    """Show validation errors untuk save operation"""
    from smartcash.ui.components.dialogs import show_warning
    
    error_messages = validation.get('errors', []) + validation.get('warnings', [])
    message = "Form validation menemukan masalah sebelum menyimpan:\n\n" + "\n".join(error_messages)
    message += "\n\nSilakan perbaiki input form sebelum menyimpan."
    
    show_warning(
        title="Validation Error - Save Config",
        message=message,
        on_close=lambda btn: None
    )

def _show_config_summary_confirmation(ui_components: Dict[str, Any]):
    """Show config summary dengan save confirmation"""
    from smartcash.ui.components.dialogs import show_confirmation
    from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
    
    # Extract current config
    current_config = extract_augmentation_config(ui_components)
    
    # Create summary message
    aug_config = current_config.get('augmentation', {})
    summary_items = [
        f"🎯 Variations: {aug_config.get('num_variations', 3)}",
        f"📊 Target Count: {aug_config.get('target_count', 500)}",
        f"🔄 Types: {', '.join(aug_config.get('types', ['combined']))}",
        f"📂 Split: {aug_config.get('target_split', 'train')}",
        f"⚖️ Balance Classes: {'Yes' if aug_config.get('balance_classes', True) else 'No'}"
    ]
    
    message = "Simpan konfigurasi dengan pengaturan berikut?\n\n" + "\n".join(summary_items)
    
    def on_confirm_save(button):
        _execute_save_config(ui_components)
    
    def on_cancel_save(button):
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, "❌ Save config dibatalkan", "info")
    
    show_confirmation(
        title="Konfirmasi Save Konfigurasi",
        message=message,
        on_confirm=on_confirm_save,
        on_cancel=on_cancel_save
    )

def _execute_save_config(ui_components: Dict[str, Any]):
    """Execute save config operation"""
    from smartcash.ui.dataset.augmentation.utils.button_manager import with_button_management
    
    @with_button_management
    def _save_config_operation(ui_components):
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
                log_to_ui(ui_components, "❌ Config handler tidak tersedia", "error")
                return
            
            # Set UI components untuk logging
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            # Execute save
            success = config_handler.save_config(ui_components)
            
            if success:
                _show_save_success_dialog(ui_components)
            else:
                from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
                log_to_ui(ui_components, "❌ Gagal menyimpan konfigurasi", "error")
                
        except Exception as e:
            from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
            log_to_ui(ui_components, f"❌ Error save config: {str(e)}", "error")
    
    _save_config_operation(ui_components)

def _execute_reset_config(ui_components: Dict[str, Any]):
    """Execute reset config operation"""
    from smartcash.ui.dataset.augmentation.utils.button_manager import with_button_management
    
    @with_button_management
    def _reset_config_operation(ui_components):
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
                log_to_ui(ui_components, "❌ Config handler tidak tersedia", "error")
                return
            
            # Set UI components untuk logging
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            # Execute reset
            success = config_handler.reset_config(ui_components)
            
            if success:
                _show_reset_success_dialog(ui_components)
            else:
                from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
                log_to_ui(ui_components, "❌ Gagal reset konfigurasi", "error")
                
        except Exception as e:
            from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
            log_to_ui(ui_components, f"❌ Error reset config: {str(e)}", "error")
    
    _reset_config_operation(ui_components)

def _show_save_success_dialog(ui_components: Dict[str, Any]):
    """Show save success dialog dengan options"""
    from smartcash.ui.components.dialogs import show_info
    
    message = "✅ Konfigurasi berhasil disimpan!\n\nKonfigurasi telah disimpan ke file dan UI telah direfresh dengan nilai terbaru."
    
    show_info(
        title="Save Config Berhasil",
        message=message,
        on_close=lambda btn: None
    )

def _show_reset_success_dialog(ui_components: Dict[str, Any]):
    """Show reset success dialog"""
    from smartcash.ui.components.dialogs import show_info
    
    message = "✅ Konfigurasi berhasil direset!\n\nSemua pengaturan telah dikembalikan ke nilai default dan UI telah diperbarui."
    
    show_info(
        title="Reset Config Berhasil", 
        message=message,
        on_close=lambda btn: None
    )