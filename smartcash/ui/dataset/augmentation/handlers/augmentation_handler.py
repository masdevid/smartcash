"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handler.py
Deskripsi: Fixed handler tanpa circular imports
"""

from typing import Dict, Any
from IPython.display import display

def handle_augmentation_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler augmentation menggunakan direct approach"""
    _execute_operation_safely(ui_components, 'augmentation', _run_augmentation_pipeline)

def handle_check_dataset_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler check dataset"""
    _execute_operation_safely(ui_components, 'check', _check_dataset_status)

def handle_cleanup_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler cleanup dengan confirmation"""
    def confirm_cleanup(confirm_button):
        _execute_operation_safely(ui_components, 'cleanup', _run_cleanup_operation)
    
    _show_confirmation_dialog(ui_components, "Konfirmasi Cleanup Dataset", 
                            "Apakah Anda yakin ingin menghapus semua file augmented?\n\n⚠️ Tindakan ini tidak dapat dibatalkan!", 
                            confirm_cleanup)

def handle_save_config_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler save config"""
    _execute_config_operation_safely(ui_components, 'save', _save_config_operation)

def handle_reset_config_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler reset config"""
    _execute_config_operation_safely(ui_components, 'reset', _reset_config_operation)

def _execute_operation_safely(ui_components: Dict[str, Any], operation_name: str, operation_func):
    """Execute operation dengan direct button state management"""
    try:
        # Get button manager directly
        button_manager = _get_or_create_button_manager(ui_components)
        
        # Check if operation can start
        can_start, message = button_manager.can_start_operation(operation_name)
        if not can_start:
            _log_to_ui_safe(ui_components, f"⚠️ {message}", 'warning')
            return
        
        # Clear logs dan reset progress
        _clear_logs_and_reset_progress(ui_components)
        
        # Use operation context
        with button_manager.operation_context(operation_name):
            operation_func(ui_components)
            _log_to_ui_safe(ui_components, f"✅ {operation_name.title()} berhasil diselesaikan", 'success')
            
    except Exception as e:
        error_msg = f"❌ {operation_name.title()} gagal: {str(e)}"
        _log_to_ui_safe(ui_components, error_msg, 'error')
        _update_status_panel_safe(ui_components, error_msg, 'error')

def _execute_config_operation_safely(ui_components: Dict[str, Any], operation_name: str, operation_func):
    """Execute config operation safely"""
    try:
        # Get button manager directly
        button_manager = _get_or_create_button_manager(ui_components)
        
        # Clear logs untuk config operations
        _clear_logs_and_reset_progress(ui_components)
        
        # Use config context (doesn't disable operation buttons)
        with button_manager.config_context(operation_name):
            operation_func(ui_components)
            success_msg = f"✅ {operation_name.title()} berhasil"
            _log_to_ui_safe(ui_components, success_msg, 'success')
            _update_status_panel_safe(ui_components, success_msg, 'success')
            
    except Exception as e:
        error_msg = f"❌ {operation_name.title()} gagal: {str(e)}"
        _log_to_ui_safe(ui_components, error_msg, 'error')
        _update_status_panel_safe(ui_components, error_msg, 'error')

def _get_or_create_button_manager(ui_components: Dict[str, Any]):
    """Get or create button manager directly tanpa import circular"""
    if 'button_state_manager' not in ui_components:
        # Import here to avoid circular dependency
        try:
            from smartcash.ui.utils.button_state_manager import create_button_state_manager
            ui_components['button_state_manager'] = create_button_state_manager(ui_components)
        except ImportError:
            # Fallback: create simple manager inline
            ui_components['button_state_manager'] = _create_simple_fallback_manager(ui_components)
    
    return ui_components['button_state_manager']

def _create_simple_fallback_manager(ui_components: Dict[str, Any]):
    """Create simple fallback manager inline untuk avoid circular import"""
    from contextlib import contextmanager
    
    class SimpleFallbackManager:
        def __init__(self, ui_components):
            self.ui_components = ui_components
        
        @contextmanager
        def operation_context(self, operation_name: str):
            try:
                # Disable operation buttons manually
                operation_buttons = ['augment_button', 'check_button', 'cleanup_button']
                disabled_buttons = []
                
                for btn_key in operation_buttons:
                    button = self.ui_components.get(btn_key)
                    if button and hasattr(button, 'disabled') and not button.disabled:
                        button.disabled = True
                        disabled_buttons.append(btn_key)
                
                yield
            finally:
                # Re-enable buttons
                for btn_key in disabled_buttons:
                    button = self.ui_components.get(btn_key)
                    if button and hasattr(button, 'disabled'):
                        button.disabled = False
        
        @contextmanager
        def config_context(self, config_operation: str):
            # Config operations don't disable buttons
            try:
                yield
            finally:
                pass
        
        def can_start_operation(self, operation_name: str):
            return True, "Simple fallback - operation allowed"
    
    return SimpleFallbackManager(ui_components)

def _clear_logs_and_reset_progress(ui_components: Dict[str, Any]):
    """Clear logs dan reset progress safely"""
    try:
        # Clear log outputs - priority order
        log_widgets = ['log_output', 'status', 'output']
        for widget_key in log_widgets:
            widget = ui_components.get(widget_key)
            if widget and hasattr(widget, 'clear_output'):
                widget.clear_output(wait=True)
                break  # Only clear first available widget
        
        # Reset progress tracker
        tracker = ui_components.get('tracker')
        if tracker and hasattr(tracker, 'reset'):
            tracker.reset()
        elif 'reset_all' in ui_components and callable(ui_components['reset_all']):
            ui_components['reset_all']()
    except Exception:
        pass

def _log_to_ui_safe(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Safe logging ke UI dengan fallback ke existing pattern"""
    try:
        # Priority 1: Use existing UI logger if available
        logger = ui_components.get('logger')
        if logger and hasattr(logger, level):
            getattr(logger, level)(message)
            return
        
        # Priority 2: Use log output widget directly (existing pattern)
        log_widget = ui_components.get('log_output') or ui_components.get('status')
        if log_widget and hasattr(log_widget, 'clear_output'):
            from IPython.display import HTML
            try:
                from smartcash.ui.utils.alert_utils import create_alert_html
                with log_widget:
                    display(HTML(create_alert_html(message, level)))
            except ImportError:
                # Fallback HTML
                color_map = {'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545'}
                color = color_map.get(level, '#007bff')
                html = f'<div style="color: {color}; margin: 2px 0;">{message}</div>'
                with log_widget:
                    display(HTML(html))
            return
        
        # Priority 3: Fallback to print with emoji
        emoji_map = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌'}
        print(f"{emoji_map.get(level, 'ℹ️')} {message}")
        
    except Exception:
        # Ultimate fallback
        print(f"LOG: {message}")

def _update_status_panel_safe(ui_components: Dict[str, Any], message: str, status_type: str):
    """Safe update status panel"""
    try:
        status_panel = ui_components.get('status_panel')
        if status_panel and hasattr(status_panel, 'value'):
            try:
                from smartcash.ui.utils.alert_utils import create_alert_html
                status_panel.value = create_alert_html(message, status_type)
            except ImportError:
                # Fallback HTML
                color_map = {'success': '#28a745', 'error': '#dc3545', 'warning': '#ffc107', 'info': '#007bff'}
                color = color_map.get(status_type, '#007bff')
                status_panel.value = f'<div style="color: {color}; padding: 8px;">{message}</div>'
    except Exception:
        pass

def _run_augmentation_pipeline(ui_components: Dict[str, Any]):
    """Execute augmentation pipeline"""
    try:
        from smartcash.dataset.augmentor.service import create_service_from_ui
        service = create_service_from_ui(ui_components)
        progress_callback = _get_progress_callback(ui_components)
        
        # Start operation progress
        _start_operation_progress(ui_components, 'augmentation')
        
        result = service.run_full_augmentation_pipeline(target_split='train', progress_callback=progress_callback)
        
        if result['status'] == 'success':
            success_msg = f"Pipeline berhasil: {result['total_files']} file → {result['final_output']}"
            _complete_operation_progress(ui_components, success_msg)
            _log_to_ui_safe(ui_components, f"✅ {success_msg}", 'success')
        else:
            error_msg = result.get('message', 'Unknown augmentation error')
            _error_operation_progress(ui_components, error_msg)
            raise Exception(error_msg)
    except ImportError as e:
        raise Exception(f"Service import error: {str(e)}")

def _check_dataset_status(ui_components: Dict[str, Any]):
    """Check dataset status"""
    try:
        from smartcash.dataset.augmentor.service import create_service_from_ui
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
        _log_to_ui_safe(ui_components, result_message, 'info')
    except ImportError as e:
        raise Exception(f"Service import error: {str(e)}")

def _run_cleanup_operation(ui_components: Dict[str, Any]):
    """Execute cleanup"""
    try:
        from smartcash.dataset.augmentor.service import create_service_from_ui
        service = create_service_from_ui(ui_components)
        progress_callback = _get_progress_callback(ui_components)
        
        # Start operation progress
        _start_operation_progress(ui_components, 'cleanup')
        
        result = service.cleanup_augmented_data(include_preprocessed=True, progress_callback=progress_callback)
        
        if result['status'] == 'success':
            success_msg = f"Cleanup berhasil: {result.get('total_deleted', 0)} file dihapus"
            _complete_operation_progress(ui_components, success_msg)
            _log_to_ui_safe(ui_components, f"✅ {success_msg}", 'success')
        else:
            error_msg = result.get('message', 'Unknown cleanup error')
            _error_operation_progress(ui_components, error_msg)
            raise Exception(error_msg)
    except ImportError as e:
        raise Exception(f"Service import error: {str(e)}")

def _save_config_operation(ui_components: Dict[str, Any]):
    """Save config operation"""
    try:
        from smartcash.ui.dataset.augmentation.handlers.config_handler import create_config_handler
        result = create_config_handler(ui_components).save_config()
        
        if result['status'] == 'error':
            raise Exception(result['message'])
        
        _log_to_ui_safe(ui_components, result['message'], result['status'])
    except ImportError as e:
        raise Exception(f"Config handler import error: {str(e)}")

def _reset_config_operation(ui_components: Dict[str, Any]):
    """Reset config operation"""
    try:
        from smartcash.ui.dataset.augmentation.handlers.config_handler import create_config_handler
        result = create_config_handler(ui_components).reset_to_default()
        
        if result['status'] == 'error':
            raise Exception(result['message'])
        
        _log_to_ui_safe(ui_components, result['message'], result['status'])
    except ImportError as e:
        raise Exception(f"Config handler import error: {str(e)}")

def _show_confirmation_dialog(ui_components: Dict[str, Any], title: str, message: str, confirm_callback):
    """Show confirmation dialog"""
    try:
        from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
        confirmation_dialog = create_confirmation_dialog(
            title=title, message=message, on_confirm=confirm_callback,
            on_cancel=lambda b: _log_to_ui_safe(ui_components, "❌ Operasi dibatalkan", 'info'), 
            danger_mode=True
        )
        
        if 'confirmation_area' in ui_components:
            ui_components['confirmation_area'].clear_output()
            with ui_components['confirmation_area']:
                display(confirmation_dialog)
    except ImportError as e:
        _log_to_ui_safe(ui_components, f"❌ Cannot show dialog: {str(e)}", 'error')

def _get_progress_callback(ui_components: Dict[str, Any]):
    """Get progress callback safely"""
    def progress_callback(step: str, current: int, total: int, message: str):
        try:
            percentage = int((current / max(1, total)) * 100) if total > 0 else current
            
            # Use existing progress tracking - priority order
            if 'update_progress' in ui_components and callable(ui_components['update_progress']):
                ui_components['update_progress'](step, percentage, message)
            elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'update'):
                ui_components['tracker'].update(step, percentage, message)
        except Exception:
            pass
    
    return progress_callback

def _start_operation_progress(ui_components: Dict[str, Any], operation: str):
    """Start operation progress safely"""
    try:
        if 'show_for_operation' in ui_components and callable(ui_components['show_for_operation']):
            ui_components['show_for_operation'](operation)
        elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'show'):
            ui_components['tracker'].show(operation)
    except Exception:
        pass

def _complete_operation_progress(ui_components: Dict[str, Any], message: str):
    """Complete operation progress safely"""
    try:
        if 'complete_operation' in ui_components and callable(ui_components['complete_operation']):
            ui_components['complete_operation'](message)
        elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'complete'):
            ui_components['tracker'].complete(message)
    except Exception:
        pass

def _error_operation_progress(ui_components: Dict[str, Any], message: str):
    """Error operation progress safely"""
    try:
        if 'error_operation' in ui_components and callable(ui_components['error_operation']):
            ui_components['error_operation'](message)
        elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'error'):
            ui_components['tracker'].error(message)
    except Exception:
        pass

def register_augmentation_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Register handlers dengan safe approach tanpa circular imports"""
    try:
        # Ensure ButtonStateManager exists tanpa circular import
        _get_or_create_button_manager(ui_components)
        
        # Handler mapping
        button_handlers = {
            'augment_button': handle_augmentation_button_click,
            'check_button': handle_check_dataset_button_click,
            'cleanup_button': handle_cleanup_button_click,
            'save_button': handle_save_config_button_click,
            'reset_button': handle_reset_config_button_click
        }
        
        # Register handlers safely
        registered_count = 0
        for button_key, handler_func in button_handlers.items():
            try:
                button = ui_components.get(button_key)
                if button and hasattr(button, 'on_click'):
                    # Clear existing handlers
                    if hasattr(button, '_click_handlers'):
                        button._click_handlers.callbacks.clear()
                    
                    # Register new handler
                    button.on_click(lambda b, h=handler_func, ui=ui_components: h(ui, b))
                    registered_count += 1
            except Exception:
                continue
        
        ui_components['registered_handlers'] = {
            'total': registered_count, 
            'handlers': list(button_handlers.keys()),
            'status': 'success' if registered_count > 0 else 'partial'
        }
        
        return ui_components
        
    except Exception as e:
        # Fallback registration
        ui_components['registered_handlers'] = {
            'total': 0, 
            'handlers': [], 
            'status': 'error',
            'error': str(e)
        }
        return ui_components