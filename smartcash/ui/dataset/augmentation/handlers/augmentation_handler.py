"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handler.py
Deskripsi: Fixed augmentation handler menggunakan shared button state manager dan proper progress flow
"""

from typing import Dict, Any
from IPython.display import display
from smartcash.dataset.augmentor.service import create_service_from_ui
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog

def handle_augmentation_button_click(ui_components: Dict[str, Any], button: Any):
    """Fixed handler menggunakan existing button state manager"""
    _execute_with_button_context(ui_components, 'augmentation', lambda: _run_augmentation_pipeline(ui_components))

def handle_check_dataset_button_click(ui_components: Dict[str, Any], button: Any):
    """Check handler using existing pattern"""
    _execute_with_button_context(ui_components, 'check', lambda: _check_dataset_status(ui_components))

def handle_cleanup_button_click(ui_components: Dict[str, Any], button: Any):
    """Cleanup handler dengan confirmation"""
    def confirm_cleanup(confirm_button):
        _execute_with_button_context(ui_components, 'cleanup', lambda: _run_cleanup_operation(ui_components))
    
    _show_confirmation_dialog(ui_components, "Konfirmasi Cleanup Dataset", 
                            "Apakah Anda yakin ingin menghapus semua file augmented?\n\n⚠️ Tindakan ini tidak dapat dibatalkan!", 
                            confirm_cleanup)

def handle_save_config_button_click(ui_components: Dict[str, Any], button: Any):
    """Save config handler"""
    _execute_with_button_context(ui_components, 'save', lambda: _save_config_operation(ui_components))

def handle_reset_config_button_click(ui_components: Dict[str, Any], button: Any):
    """Reset config handler"""
    _execute_with_button_context(ui_components, 'reset', lambda: _reset_config_operation(ui_components))

# Consolidated DRY helper functions
def _execute_with_button_context(ui_components: Dict[str, Any], operation: str, operation_func):
    """One-liner execute operation dengan existing button context"""
    from smartcash.ui.utils.button_state_manager import get_button_state_manager
    button_manager = get_button_state_manager(ui_components)
    
    try:
        with button_manager.operation_context(operation):
            operation_func()
    except Exception as e:
        _log_to_ui(ui_components, f"Error {operation}: {str(e)}", 'error')

def _run_augmentation_pipeline(ui_components: Dict[str, Any]):
    """Execute augmentation pipeline dengan existing pattern"""
    from smartcash.ui.dataset.augmentation.handlers.config_handler import extract_config
    
    service = create_service_from_ui(ui_components)
    progress_callback = _get_existing_progress_callback(ui_components)
    
    result = service.run_full_augmentation_pipeline(target_split='train', progress_callback=progress_callback)
    
    if result['status'] == 'success':
        _log_to_ui(ui_components, f"Pipeline berhasil: {result['total_files']} file → {result['final_output']}", 'success')
    else:
        _log_to_ui(ui_components, f"Augmentasi gagal: {result.get('message', 'Unknown error')}", 'error')

def _check_dataset_status(ui_components: Dict[str, Any]):
    """Check dataset status operation"""
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
    progress_callback = _get_existing_progress_callback(ui_components)
    
    result = service.cleanup_augmented_data(include_preprocessed=True, progress_callback=progress_callback)
    
    if result['status'] == 'success':
        _log_to_ui(ui_components, f"Cleanup berhasil: {result.get('total_deleted', 0)} file dihapus", 'success')
    else:
        _log_to_ui(ui_components, f"Cleanup gagal: {result.get('message', 'Unknown error')}", 'error')

def _save_config_operation(ui_components: Dict[str, Any]):
    """Save config operation"""
    from smartcash.ui.dataset.augmentation.handlers.config_handler import create_config_handler
    result = create_config_handler(ui_components).save_config()
    _log_to_ui(ui_components, result['message'], result['status'])

def _reset_config_operation(ui_components: Dict[str, Any]):
    """Reset config operation"""
    from smartcash.ui.dataset.augmentation.handlers.config_handler import create_config_handler
    result = create_config_handler(ui_components).reset_to_default()
    _log_to_ui(ui_components, result['message'], result['status'])

def _show_confirmation_dialog(ui_components: Dict[str, Any], title: str, message: str, confirm_callback):
    """Show confirmation dialog using existing pattern"""
    confirmation_dialog = create_confirmation_dialog(
        title=title, message=message, on_confirm=confirm_callback,
        on_cancel=lambda b: _log_to_ui(ui_components, "Operasi dibatalkan", 'info'), danger_mode=True
    )
    
    if 'confirmation_area' in ui_components:
        ui_components['confirmation_area'].clear_output()
        with ui_components['confirmation_area']:
            display(confirmation_dialog)

def _get_existing_progress_callback(ui_components: Dict[str, Any]):
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
            
            # Reduced logging frequency to prevent flooding
            if current % max(1, total // 5) == 0 or current == total:
                logger = ui_components.get('logger')
                logger and hasattr(logger, 'debug') and logger.debug(f"📊 {step}: {message} ({percentage}%)")
                    
        except Exception:
            pass
    
    return progress_callback

# Helper untuk unified logging tanpa flooding
def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """One-liner logging ke UI tanpa flooding"""
    ui_components.get('logger') and getattr(ui_components['logger'], level, lambda x: None)(message)

def _handle_error(ui_components: Dict[str, Any], button_manager, progress_handler, error_msg: str):
    """One-liner error handler"""
    button_manager.restore_button_state('augment_button')
    progress_handler.error_operation(error_msg)
    _log_to_ui(ui_components, f"❌ {error_msg}", 'error')

def register_augmentation_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Register handlers menggunakan existing implementation"""
    # Use existing handler registration pattern
    button_handlers = {
        'augment_button': handle_augmentation_button_click,
        'check_button': handle_check_dataset_button_click,
        'cleanup_button': handle_cleanup_button_click,
        'save_button': handle_save_config_button_click,
        'reset_button': handle_reset_config_button_click
    }
    
    registered_count = 0
    for button_key, handler_func in button_handlers.items():
        button = ui_components.get(button_key)
        if button and hasattr(button, 'on_click'):
            button.on_click(lambda b, h=handler_func, ui=ui_components: h(ui, b))
            registered_count += 1
    
    ui_components['registered_handlers'] = {'total': registered_count, 'handlers': list(button_handlers.keys())}
    return ui_components