"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handler.py
Deskripsi: Fixed handler menggunakan existing ButtonStateManager dan progress components
"""

from typing import Dict, Any
from IPython.display import display
from smartcash.dataset.augmentor.service import create_service_from_ui
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.ui.utils.button_state_manager import get_button_state_manager, get_operation_context, get_config_context

def handle_augmentation_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler menggunakan generic ButtonStateManager"""
    _execute_operation_with_state_manager(ui_components, 'augmentation', _run_augmentation_pipeline)

def handle_check_dataset_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler menggunakan generic ButtonStateManager"""
    _execute_operation_with_state_manager(ui_components, 'check', _check_dataset_status)

def handle_cleanup_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler dengan confirmation menggunakan generic ButtonStateManager"""
    def confirm_cleanup(confirm_button):
        _execute_operation_with_state_manager(ui_components, 'cleanup', _run_cleanup_operation)
    
    _show_confirmation_dialog(ui_components, "Konfirmasi Cleanup Dataset", 
                            "Apakah Anda yakin ingin menghapus semua file augmented?\n\n⚠️ Tindakan ini tidak dapat dibatalkan!", 
                            confirm_cleanup)

def handle_save_config_button_click(ui_components: Dict[str, Any], button: Any):
    """Config handler menggunakan generic ButtonStateManager"""
    _execute_config_operation_with_state_manager(ui_components, 'save', _save_config_operation)

def handle_reset_config_button_click(ui_components: Dict[str, Any], button: Any):
    """Config handler menggunakan generic ButtonStateManager"""
    _execute_config_operation_with_state_manager(ui_components, 'reset', _reset_config_operation)

def _execute_operation_with_state_manager(ui_components: Dict[str, Any], operation_name: str, operation_func):
    """Execute operation menggunakan generic ButtonStateManager context"""
    button_manager = get_button_state_manager(ui_components)
    
    # Check if operation can start (generic across all modules)
    can_start, message = button_manager.can_start_operation(operation_name)
    if not can_start:
        _log_to_ui(ui_components, f"⚠️ {message}", 'warning')
        return
    
    # Clear logs dan reset progress
    _clear_logs_and_reset_progress(ui_components)
    
    # Use generic operation context
    with button_manager.operation_context(operation_name):
        try:
            operation_func(ui_components)
            _log_to_ui(ui_components, f"✅ {operation_name.title()} berhasil diselesaikan", 'success')
        except Exception as e:
            error_msg = f"❌ {operation_name.title()} gagal: {str(e)}"
            _log_to_ui(ui_components, error_msg, 'error')
            raise

def _execute_config_operation_with_state_manager(ui_components: Dict[str, Any], operation_name: str, operation_func):
    """Execute config operation menggunakan generic config context (tidak disable operation buttons)"""
    button_manager = get_button_state_manager(ui_components)
    
    # Clear logs saja untuk config operations
    _clear_logs_and_reset_progress(ui_components)
    
    # Use generic config context (lightweight, doesn't disable operation buttons)
    with button_manager.config_context(operation_name):
        try:
            operation_func(ui_components)
            _update_status_panel(ui_components, f"✅ {operation_name.title()} berhasil", 'success')
        except Exception as e:
            error_msg = f"❌ {operation_name.title()} gagal: {str(e)}"
            _log_to_ui(ui_components, error_msg, 'error')
            _update_status_panel(ui_components, error_msg, 'error')

def _clear_logs_and_reset_progress(ui_components: Dict[str, Any]):
    """One-liner clear logs dan reset progress menggunakan existing components"""
    # Clear log outputs menggunakan existing pattern
    log_widgets = ['log_output', 'status', 'output']
    [getattr(ui_components.get(widget), 'clear_output', lambda **kw: None)(wait=True) 
     for widget in log_widgets if ui_components.get(widget) and hasattr(ui_components.get(widget), 'clear_output')]
    
    # Reset progress menggunakan existing tracker
    tracker = ui_components.get('tracker')
    if tracker and hasattr(tracker, 'reset'):
        tracker.reset()
    elif 'reset_all' in ui_components:
        ui_components['reset_all']()

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str):
    """One-liner update status panel menggunakan existing alert utils"""
    status_panel = ui_components.get('status_panel')
    if status_panel and hasattr(status_panel, 'value'):
        from smartcash.ui.utils.alert_utils import create_alert_html
        status_panel.value = create_alert_html(message, status_type)

def _run_augmentation_pipeline(ui_components: Dict[str, Any]):
    """Execute augmentation pipeline menggunakan existing service pattern"""
    service = create_service_from_ui(ui_components)
    progress_callback = _get_progress_callback(ui_components)
    
    # Start operation progress
    _start_operation_progress(ui_components, 'augmentation')
    
    result = service.run_full_augmentation_pipeline(target_split='train', progress_callback=progress_callback)
    
    if result['status'] == 'success':
        _complete_operation_progress(ui_components, f"Pipeline berhasil: {result['total_files']} file → {result['final_output']}")
        _log_to_ui(ui_components, f"✅ Pipeline berhasil: {result['total_files']} file → {result['final_output']}", 'success')
    else:
        _error_operation_progress(ui_components, result.get('message', 'Unknown augmentation error'))
        raise Exception(result.get('message', 'Unknown augmentation error'))

def _check_dataset_status(ui_components: Dict[str, Any]):
    """Check dataset status menggunakan existing service pattern"""
    service = create_service_from_ui(ui_components)
    
    # Start operation progress
    _start_operation_progress(ui_components, 'check')
    
    status = service.get_augmentation_status()
    
    status_lines = [
        f"📁 Raw: {'✅' if status.get('raw_dataset', {}).get('exists') else '❌'} ({status.get('raw_dataset', {}).get('total_images', 0)} img)",
        f"🔄 Aug: {'✅' if status.get('augmented_dataset', {}).get('exists') else '❌'} ({status.get('augmented_dataset', {}).get('total_images', 0)} files)",
        f"📊 Prep: {'✅' if status.get('preprocessed_dataset', {}).get('exists') else '❌'} ({status.get('preprocessed_dataset', {}).get('total_files', 0)} files)"
    ]
    
    result_message = "📊 Status Dataset:\n" + "\n".join(status_lines)
    _complete_operation_progress(ui_components, "Check dataset selesai")
    _log_to_ui(ui_components, result_message, 'info')

def _run_cleanup_operation(ui_components: Dict[str, Any]):
    """Execute cleanup menggunakan existing service pattern"""
    service = create_service_from_ui(ui_components)
    progress_callback = _get_progress_callback(ui_components)
    
    # Start operation progress
    _start_operation_progress(ui_components, 'cleanup')
    
    result = service.cleanup_augmented_data(include_preprocessed=True, progress_callback=progress_callback)
    
    if result['status'] == 'success':
        _complete_operation_progress(ui_components, f"Cleanup berhasil: {result.get('total_deleted', 0)} file dihapus")
        _log_to_ui(ui_components, f"✅ Cleanup berhasil: {result.get('total_deleted', 0)} file dihapus", 'success')
    else:
        _error_operation_progress(ui_components, result.get('message', 'Unknown cleanup error'))
        raise Exception(result.get('message', 'Unknown cleanup error'))

def _save_config_operation(ui_components: Dict[str, Any]):
    """Save config menggunakan existing config handler"""
    from smartcash.ui.dataset.augmentation.handlers.config_handler import create_config_handler
    result = create_config_handler(ui_components).save_config()
    
    if result['status'] == 'error':
        raise Exception(result['message'])
    
    _log_to_ui(ui_components, result['message'], result['status'])

def _reset_config_operation(ui_components: Dict[str, Any]):
    """Reset config menggunakan existing config handler"""
    from smartcash.ui.dataset.augmentation.handlers.config_handler import create_config_handler
    result = create_config_handler(ui_components).reset_to_default()
    
    if result['status'] == 'error':
        raise Exception(result['message'])
    
    _log_to_ui(ui_components, result['message'], result['status'])

def _show_confirmation_dialog(ui_components: Dict[str, Any], title: str, message: str, confirm_callback):
    """Show confirmation dialog menggunakan existing component"""
    confirmation_dialog = create_confirmation_dialog(
        title=title, message=message, on_confirm=confirm_callback,
        on_cancel=lambda b: _log_to_ui(ui_components, "❌ Operasi dibatalkan", 'info'), danger_mode=True
    )
    
    if 'confirmation_area' in ui_components:
        ui_components['confirmation_area'].clear_output()
        with ui_components['confirmation_area']:
            display(confirmation_dialog)

def _get_progress_callback(ui_components: Dict[str, Any]):
    """Get progress callback menggunakan existing progress components"""
    def progress_callback(step: str, current: int, total: int, message: str):
        try:
            percentage = int((current / max(1, total)) * 100) if total > 0 else current
            
            # Use existing progress tracking methods - priority order
            if 'update_progress' in ui_components:
                ui_components['update_progress'](step, percentage, message)
            elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'update'):
                ui_components['tracker'].update(step, percentage, message)
            
        except Exception:
            pass
    
    return progress_callback

def _start_operation_progress(ui_components: Dict[str, Any], operation: str):
    """Start operation progress menggunakan existing progress components"""
    try:
        if 'show_for_operation' in ui_components:
            ui_components['show_for_operation'](operation)
        elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'show'):
            ui_components['tracker'].show(operation)
    except Exception:
        pass

def _complete_operation_progress(ui_components: Dict[str, Any], message: str):
    """Complete operation progress menggunakan existing progress components"""
    try:
        if 'complete_operation' in ui_components:
            ui_components['complete_operation'](message)
        elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'complete'):
            ui_components['tracker'].complete(message)
    except Exception:
        pass

def _error_operation_progress(ui_components: Dict[str, Any], message: str):
    """Error operation progress menggunakan existing progress components"""
    try:
        if 'error_operation' in ui_components:
            ui_components['error_operation'](message)
        elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'error'):
            ui_components['tracker'].error(message)
    except Exception:
        pass

def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """One-liner logging menggunakan existing UI logger"""
    logger = ui_components.get('logger')
    if logger and hasattr(logger, level):
        getattr(logger, level)(message)
    else:
        # Fallback ke existing log components
        log_widget = ui_components.get('log_output') or ui_components.get('status')
        if log_widget and hasattr(log_widget, 'clear_output'):
            from IPython.display import HTML
            from smartcash.ui.utils.alert_utils import create_alert_html
            with log_widget:
                display(HTML(create_alert_html(message, level)))

def register_augmentation_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Register handlers menggunakan generic ButtonStateManager approach"""
    # Ensure generic ButtonStateManager exists
    get_button_state_manager(ui_components)
    
    # Generic handler registration approach
    button_handlers = {
        'augment_button': handle_augmentation_button_click,
        'check_button': handle_check_dataset_button_click,
        'cleanup_button': handle_cleanup_button_click,
        'save_button': handle_save_config_button_click,
        'reset_button': handle_reset_config_button_click
    }
    
    # Register handlers menggunakan generic approach
    registered_count = 0
    for button_key, handler_func in button_handlers.items():
        button = ui_components.get(button_key)
        if button and hasattr(button, 'on_click'):
            try:
                # Clear existing handlers first
                button._click_handlers.callbacks.clear()
                # Register new handler
                button.on_click(lambda b, h=handler_func, ui=ui_components: h(ui, b))
                registered_count += 1
            except Exception:
                pass
    
    ui_components['registered_handlers'] = {'total': registered_count, 'handlers': list(button_handlers.keys())}
    return ui_components