"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handler.py
Deskripsi: Fixed handler dengan consolidated operation wrapper dan proper button state manager initialization
"""

from typing import Dict, Any, Callable, Optional
from IPython.display import display
from smartcash.dataset.augmentor.service import create_service_from_ui
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog

# Consolidated operation wrapper dengan null-safe manager initialization
def _safe_execute_operation(ui_components: Dict[str, Any], operation_name: str, operation_func: Callable) -> None:
    """One-liner safe operation execution dengan proper button state manager"""
    try:
        # Get or create button state manager dengan null safety
        manager = ui_components.get('button_state_manager') or _ensure_button_manager(ui_components)
        
        # Execute dengan context jika manager valid
        if manager and hasattr(manager, 'operation_context'):
            with manager.operation_context(operation_name):
                operation_func()
        else:
            # Fallback execution tanpa context
            _log_ui(ui_components, f"🔓 Executing {operation_name} without button lock", 'warning')
            operation_func()
            
    except Exception as e:
        _log_ui(ui_components, f"❌ Error {operation_name}: {str(e)}", 'error')

def _ensure_button_manager(ui_components: Dict[str, Any]) -> Optional[Any]:
    """One-liner button manager initialization dengan fallback"""
    try:
        from smartcash.ui.utils.button_state_manager import get_button_state_manager
        manager = get_button_state_manager(ui_components)
        ui_components['button_state_manager'] = manager
        return manager
    except Exception as e:
        _log_ui(ui_components, f"⚠️ Button manager fallback: {str(e)}", 'debug')
        return None

# Consolidated handlers menggunakan safe wrapper
def handle_augmentation_button_click(ui_components: Dict[str, Any], button: Any):
    """Fixed augmentation handler"""
    _safe_execute_operation(ui_components, 'augmentation', 
                           lambda: _run_augmentation_pipeline(ui_components))

def handle_check_dataset_button_click(ui_components: Dict[str, Any], button: Any):
    """Fixed check handler"""
    _safe_execute_operation(ui_components, 'check', 
                           lambda: _check_dataset_status(ui_components))

def handle_cleanup_button_click(ui_components: Dict[str, Any], button: Any):
    """Fixed cleanup handler dengan confirmation"""
    _show_confirmation_dialog(ui_components, "Konfirmasi Cleanup Dataset", 
                            "Apakah Anda yakin ingin menghapus semua file augmented?\n\n⚠️ Tindakan ini tidak dapat dibatalkan!",
                            lambda confirm_btn: _safe_execute_operation(ui_components, 'cleanup', 
                                                                      lambda: _run_cleanup_operation(ui_components)))

def handle_save_config_button_click(ui_components: Dict[str, Any], button: Any):
    """Fixed save config handler"""
    _safe_execute_operation(ui_components, 'save', 
                           lambda: _save_config_operation(ui_components))

def handle_reset_config_button_click(ui_components: Dict[str, Any], button: Any):
    """Fixed reset config handler"""
    _safe_execute_operation(ui_components, 'reset', 
                           lambda: _reset_config_operation(ui_components))

# Consolidated operation implementations
def _run_augmentation_pipeline(ui_components: Dict[str, Any]) -> None:
    """Consolidated augmentation pipeline dengan progress callback"""
    service = create_service_from_ui(ui_components)
    progress_cb = _create_progress_callback(ui_components)
    
    result = service.run_full_augmentation_pipeline(target_split='train', progress_callback=progress_cb)
    status_msg = f"Pipeline berhasil: {result['total_files']} file → {result['final_output']}" if result['status'] == 'success' else f"Augmentasi gagal: {result.get('message', 'Unknown error')}"
    _log_ui(ui_components, status_msg, result['status'])

def _check_dataset_status(ui_components: Dict[str, Any]) -> None:
    """Consolidated dataset status check"""
    service, status = create_service_from_ui(ui_components), create_service_from_ui(ui_components).get_augmentation_status()
    
    # One-liner status formatting
    status_lines = [f"{icon} {name}: {'✅' if data.get('exists') else '❌'} ({data.get('total_images', data.get('total_files', 0))} files)"
                   for (icon, name, data) in [('📁', 'Raw', status.get('raw_dataset', {})), 
                                              ('🔄', 'Aug', status.get('augmented_dataset', {})),
                                              ('📊', 'Prep', status.get('preprocessed_dataset', {}))]]
    
    _log_ui(ui_components, "📊 Status Dataset:\n" + "\n".join(status_lines), 'info')

def _run_cleanup_operation(ui_components: Dict[str, Any]) -> None:
    """Consolidated cleanup operation"""
    service = create_service_from_ui(ui_components)
    progress_cb = _create_progress_callback(ui_components)
    
    result = service.cleanup_augmented_data(include_preprocessed=True, progress_callback=progress_cb)
    status_msg = f"Cleanup berhasil: {result.get('total_deleted', 0)} file dihapus" if result['status'] == 'success' else f"Cleanup gagal: {result.get('message', 'Unknown error')}"
    _log_ui(ui_components, status_msg, result['status'])

def _save_config_operation(ui_components: Dict[str, Any]) -> None:
    """Consolidated save config operation"""
    from smartcash.ui.dataset.augmentation.handlers.config_handler import create_config_handler
    result = create_config_handler(ui_components).save_config()
    _log_ui(ui_components, result['message'], result['status'])

def _reset_config_operation(ui_components: Dict[str, Any]) -> None:
    """Consolidated reset config operation"""
    from smartcash.ui.dataset.augmentation.handlers.config_handler import create_config_handler
    result = create_config_handler(ui_components).reset_to_default()
    _log_ui(ui_components, result['message'], result['status'])

# Consolidated utilities
def _create_progress_callback(ui_components: Dict[str, Any]) -> Callable:
    """One-liner progress callback creation"""
    tracker = ui_components.get('tracker')
    logger = ui_components.get('logger')
    
    def progress_cb(step: str, current: int, total: int, message: str):
        percentage = int((current / max(1, total)) * 100) if total > 0 else current
        # Update progress dengan reduced frequency
        tracker and hasattr(tracker, 'update') and current % max(1, total // 5) == 0 and tracker.update(step, percentage, message)
        # Log hanya pada milestone
        logger and hasattr(logger, 'debug') and current % max(1, total // 10) == 0 and logger.debug(f"📊 {step}: {message} ({percentage}%)")
    
    return progress_cb

def _show_confirmation_dialog(ui_components: Dict[str, Any], title: str, message: str, confirm_callback: Callable) -> None:
    """One-liner confirmation dialog dengan existing pattern"""
    confirmation_dialog = create_confirmation_dialog(title=title, message=message, on_confirm=confirm_callback,
                                                   on_cancel=lambda b: _log_ui(ui_components, "Operasi dibatalkan", 'info'), danger_mode=True)
    
    # Display di confirmation area jika tersedia
    confirmation_area = ui_components.get('confirmation_area')
    confirmation_area and hasattr(confirmation_area, 'clear_output') and confirmation_area.clear_output()
    confirmation_area and hasattr(confirmation_area, 'clear_output') and display(confirmation_dialog) or None

def _log_ui(ui_components: Dict[str, Any], message: str, level: str = 'info') -> None:
    """One-liner UI logging dengan null safety"""
    logger = ui_components.get('logger')
    logger and hasattr(logger, level) and getattr(logger, level)(message)

def register_augmentation_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Consolidated handler registration dengan proper button manager initialization"""
    # Ensure button manager exists
    _ensure_button_manager(ui_components)
    
    # One-liner handler mapping
    handlers = {btn: handler for btn, handler in [
        ('augment_button', handle_augmentation_button_click),
        ('check_button', handle_check_dataset_button_click), 
        ('cleanup_button', handle_cleanup_button_click),
        ('save_button', handle_save_config_button_click),
        ('reset_button', handle_reset_config_button_click)
    ]}
    
    # Register handlers dengan null safety
    registered = sum(1 for btn_key, handler_func in handlers.items() 
                    if (button := ui_components.get(btn_key)) and hasattr(button, 'on_click') 
                    and button.on_click(lambda b, h=handler_func, ui=ui_components: h(ui, b)) or True)
    
    ui_components['registered_handlers'] = {'total': registered, 'handlers': list(handlers.keys())}
    _log_ui(ui_components, f"✅ {registered} handlers berhasil didaftarkan", 'info')
    
    return ui_components