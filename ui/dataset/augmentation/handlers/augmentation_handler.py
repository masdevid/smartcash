"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handler.py
Deskripsi: Fixed augmentation handler tanpa threading dan dengan button lock yang proper
"""

from typing import Dict, Any
from IPython.display import display
from smartcash.dataset.augmentor.service import create_service_from_ui
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog

def handle_augmentation_button_click(ui_components: Dict[str, Any], button: Any):
    """Fixed handler dengan proper button state management tanpa threading"""
    _execute_safe_operation(ui_components, 'augmentation', lambda: _run_augmentation_pipeline(ui_components))

def handle_check_dataset_button_click(ui_components: Dict[str, Any], button: Any):
    """Fixed check handler tanpa threading - langsung execute"""
    _execute_safe_operation(ui_components, 'check', lambda: _check_dataset_status(ui_components))

def handle_cleanup_button_click(ui_components: Dict[str, Any], button: Any):
    """Fixed cleanup handler dengan confirmation"""
    def confirm_cleanup(confirm_button):
        _execute_safe_operation(ui_components, 'cleanup', lambda: _run_cleanup_operation(ui_components))
    
    _show_confirmation_dialog(ui_components, "Konfirmasi Cleanup Dataset", 
                            "Apakah Anda yakin ingin menghapus semua file augmented?\n\n⚠️ Tindakan ini tidak dapat dibatalkan!", 
                            confirm_cleanup)

def handle_save_config_button_click(ui_components: Dict[str, Any], button: Any):
    """Save config handler dengan proper state management"""
    _execute_safe_operation(ui_components, 'save', lambda: _save_config_operation(ui_components))

def handle_reset_config_button_click(ui_components: Dict[str, Any], button: Any):
    """Reset config handler dengan proper state management"""
    _execute_safe_operation(ui_components, 'reset', lambda: _reset_config_operation(ui_components))

# One-liner safe operation executor
def _execute_safe_operation(ui_components: Dict[str, Any], operation: str, operation_func):
    """One-liner safe operation dengan proper error handling dan state management"""
    try:
        # Set button state
        _set_button_processing_state(ui_components, operation)
        
        # Execute operation
        operation_func()
        
        # Success state
        _set_button_success_state(ui_components, operation)
        
    except Exception as e:
        _handle_operation_error(ui_components, operation, str(e))
    finally:
        # Always restore button state after delay
        _schedule_button_restore(ui_components, operation)

def _run_augmentation_pipeline(ui_components: Dict[str, Any]):
    """Execute augmentation pipeline dengan existing pattern"""
    service = create_service_from_ui(ui_components)
    progress_callback = _get_progress_callback(ui_components)
    
    result = service.run_full_augmentation_pipeline(target_split='train', progress_callback=progress_callback)
    
    if result['status'] == 'success':
        _log_to_ui(ui_components, f"✅ Pipeline berhasil: {result['total_files']} file → {result['final_output']}", 'success')
    else:
        raise Exception(result.get('message', 'Unknown augmentation error'))

def _check_dataset_status(ui_components: Dict[str, Any]):
    """Check dataset status operation - langsung execute tanpa threading"""
    service = create_service_from_ui(ui_components)
    status = service.get_augmentation_status()
    
    status_lines = [
        f"📁 Raw: {'✅' if status.get('raw_dataset', {}).get('exists') else '❌'} ({status.get('raw_dataset', {}).get('total_images', 0)} img)",
        f"🔄 Aug: {'✅' if status.get('augmented_dataset', {}).get('exists') else '❌'} ({status.get('augmented_dataset', {}).get('total_images', 0)} files)",
        f"📊 Prep: {'✅' if status.get('preprocessed_dataset', {}).get('exists') else '❌'} ({status.get('preprocessed_dataset', {}).get('total_files', 0)} files)"
    ]
    
    _log_to_ui(ui_components, "📊 Status Dataset:\n" + "\n".join(status_lines), 'info')

def _run_cleanup_operation(ui_components: Dict[str, Any]):
    """Execute cleanup operation"""
    service = create_service_from_ui(ui_components)
    progress_callback = _get_progress_callback(ui_components)
    
    result = service.cleanup_augmented_data(include_preprocessed=True, progress_callback=progress_callback)
    
    if result['status'] == 'success':
        _log_to_ui(ui_components, f"✅ Cleanup berhasil: {result.get('total_deleted', 0)} file dihapus", 'success')
    else:
        raise Exception(result.get('message', 'Unknown cleanup error'))

def _save_config_operation(ui_components: Dict[str, Any]):
    """Save config operation"""
    from smartcash.ui.dataset.augmentation.handlers.config_handler import create_config_handler
    result = create_config_handler(ui_components).save_config()
    
    if result['status'] == 'error':
        raise Exception(result['message'])
    
    _log_to_ui(ui_components, result['message'], result['status'])

def _reset_config_operation(ui_components: Dict[str, Any]):
    """Reset config operation"""
    from smartcash.ui.dataset.augmentation.handlers.config_handler import create_config_handler
    result = create_config_handler(ui_components).reset_to_default()
    
    if result['status'] == 'error':
        raise Exception(result['message'])
    
    _log_to_ui(ui_components, result['message'], result['status'])

def _show_confirmation_dialog(ui_components: Dict[str, Any], title: str, message: str, confirm_callback):
    """Show confirmation dialog using existing pattern"""
    confirmation_dialog = create_confirmation_dialog(
        title=title, message=message, on_confirm=confirm_callback,
        on_cancel=lambda b: _log_to_ui(ui_components, "❌ Operasi dibatalkan", 'info'), danger_mode=True
    )
    
    if 'confirmation_area' in ui_components:
        ui_components['confirmation_area'].clear_output()
        with ui_components['confirmation_area']:
            display(confirmation_dialog)

def _get_progress_callback(ui_components: Dict[str, Any]):
    """Get progress callback using existing components"""
    def progress_callback(step: str, current: int, total: int, message: str):
        try:
            percentage = int((current / max(1, total)) * 100) if total > 0 else current
            
            # Use existing progress tracking methods
            tracker = ui_components.get('tracker')
            if tracker and hasattr(tracker, 'update'):
                tracker.update(step, percentage, message)
            elif 'update_progress' in ui_components:
                ui_components['update_progress'](step, percentage, message)
            
        except Exception:
            pass
    
    return progress_callback

# Button state management helpers
def _set_button_processing_state(ui_components: Dict[str, Any], operation: str):
    """One-liner set button to processing state"""
    button_key = _get_button_key_for_operation(operation)
    button = ui_components.get(button_key)
    
    button and [
        setattr(button, 'disabled', True),
        setattr(button, 'description', f"⏳ {_get_processing_text(operation)}"),
        setattr(button, 'button_style', 'warning')
    ]

def _set_button_success_state(ui_components: Dict[str, Any], operation: str):
    """One-liner set button to success state"""
    button_key = _get_button_key_for_operation(operation)
    button = ui_components.get(button_key)
    
    button and [
        setattr(button, 'description', f"✅ {_get_success_text(operation)}"),
        setattr(button, 'button_style', 'success')
    ]

def _schedule_button_restore(ui_components: Dict[str, Any], operation: str):
    """Schedule button restore tanpa threading - menggunakan simple timer"""
    import time
    time.sleep(2)  # Simple delay tanpa threading
    _restore_button_state(ui_components, operation)

def _restore_button_state(ui_components: Dict[str, Any], operation: str):
    """One-liner restore button to original state"""
    button_key = _get_button_key_for_operation(operation)
    button = ui_components.get(button_key)
    
    button and [
        setattr(button, 'disabled', False),
        setattr(button, 'description', _get_original_text(operation)),
        setattr(button, 'button_style', _get_original_style(operation))
    ]

def _handle_operation_error(ui_components: Dict[str, Any], operation: str, error_msg: str):
    """Handle operation error dengan proper logging"""
    button_key = _get_button_key_for_operation(operation)
    button = ui_components.get(button_key)
    
    button and [
        setattr(button, 'description', f"❌ Error!"),
        setattr(button, 'button_style', 'danger')
    ]
    
    _log_to_ui(ui_components, f"❌ {operation.title()} gagal: {error_msg}", 'error')

# One-liner mapping helpers
_get_button_key_for_operation = lambda op: {'augmentation': 'augment_button', 'check': 'check_button', 'cleanup': 'cleanup_button', 'save': 'save_button', 'reset': 'reset_button'}.get(op, 'augment_button')
_get_processing_text = lambda op: {'augmentation': 'Processing...', 'check': 'Checking...', 'cleanup': 'Cleaning...', 'save': 'Saving...', 'reset': 'Resetting...'}.get(op, 'Processing...')
_get_success_text = lambda op: {'augmentation': 'Selesai!', 'check': 'Checked!', 'cleanup': 'Cleaned!', 'save': 'Saved!', 'reset': 'Reset!'}.get(op, 'Done!')
_get_original_text = lambda op: {'augmentation': '🚀 Run Augmentation', 'check': '🔍 Check Dataset', 'cleanup': '🧹 Cleanup Dataset', 'save': 'Simpan', 'reset': 'Reset'}.get(op, 'Action')
_get_original_style = lambda op: {'augmentation': '', 'check': 'info', 'cleanup': 'warning', 'save': 'primary', 'reset': ''}.get(op, '')

# Helper untuk unified logging
def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """One-liner logging ke UI"""
    ui_components.get('logger') and getattr(ui_components['logger'], level, lambda x: None)(message)

def register_augmentation_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Register handlers dengan mapping yang clear"""
    # One-liner handler registration
    button_handlers = {
        'augment_button': handle_augmentation_button_click,
        'check_button': handle_check_dataset_button_click,
        'cleanup_button': handle_cleanup_button_click,
        'save_button': handle_save_config_button_click,
        'reset_button': handle_reset_config_button_click
    }
    
    # Register handlers dengan one-liner pattern
    registered_count = sum(1 for button_key, handler_func in button_handlers.items() 
                          if (button := ui_components.get(button_key)) and hasattr(button, 'on_click') 
                          and not button.on_click(lambda b, h=handler_func, ui=ui_components: h(ui, b)))
    
    ui_components['registered_handlers'] = {'total': len(button_handlers), 'handlers': list(button_handlers.keys())}
    return ui_components