"""
File: smartcash/ui/utils/safe_handler_utils.py
Deskripsi: Utilitas untuk safe handler pattern dengan null safety dan error handling terpusat
"""

from typing import Dict, Any, Callable, Optional, Tuple
from contextlib import contextmanager
from functools import wraps

def safe_handler_wrapper(operation_name: str):
    """
    Decorator untuk membuat handler yang aman dengan null safety pattern.
    
    Args:
        operation_name: Nama operasi untuk button state management
        
    Returns:
        Decorated function dengan null safety dan button state management
    """
    def decorator(handler_func: Callable):
        @wraps(handler_func)
        def wrapper(ui_components: Dict[str, Any], button: Any = None, *args, **kwargs):
            # Setup logger dengan null safety
            ui_logger = create_safe_logger(ui_components, handler_func.__name__)
            
            # Ensure button state manager
            if not ensure_button_state_manager_safe(ui_components, ui_logger):
                return
            
            # Get button state manager dan cek operation
            button_state_manager = get_button_state_manager_safe(ui_components, ui_logger)
            if not button_state_manager:
                return
            
            # Cek apakah operation bisa dimulai
            if not check_operation_can_start(button_state_manager, operation_name, ui_logger):
                return
            
            # Jalankan handler dengan context manager
            try:
                with button_state_manager.operation_context(operation_name):
                    return handler_func(ui_components, button, ui_logger, *args, **kwargs)
            except Exception as e:
                ui_logger.error(f"❌ Error dalam {operation_name}: {str(e)}")
                safe_update_status_panel(ui_components, f"❌ Error: {str(e)}", "error")
        
        return wrapper
    return decorator

def create_safe_logger(ui_components: Dict[str, Any], context: str = "handler"):
    """Create logger dengan fallback yang aman."""
    try:
        from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
        return create_ui_logger_bridge(ui_components, context)
    except Exception:
        return create_fallback_logger()

def create_fallback_logger():
    """Create fallback logger untuk emergency cases."""
    class FallbackLogger:
        def info(self, msg): print(f"ℹ️ {msg}")
        def success(self, msg): print(f"✅ {msg}")
        def warning(self, msg): print(f"⚠️ {msg}")
        def error(self, msg): print(f"❌ {msg}")
        def debug(self, msg): print(f"🔍 {msg}")
        def critical(self, msg): print(f"🔥 {msg}")
    
    return FallbackLogger()

def ensure_button_state_manager_safe(ui_components: Dict[str, Any], ui_logger) -> bool:
    """Ensure button state manager dengan safe error handling."""
    try:
        from smartcash.ui.utils.button_state_manager import ensure_button_state_manager
        result = ensure_button_state_manager(ui_components)
        if not result:
            ui_logger.error("❌ Gagal inisialisasi button state manager")
        return result
    except Exception as e:
        ui_logger.error(f"❌ Error ensure button state manager: {str(e)}")
        return False

def get_button_state_manager_safe(ui_components: Dict[str, Any], ui_logger):
    """Get button state manager dengan safe error handling."""
    try:
        from smartcash.ui.utils.button_state_manager import get_button_state_manager
        return get_button_state_manager(ui_components)
    except Exception as e:
        ui_logger.error(f"❌ Error get button state manager: {str(e)}")
        return None

def check_operation_can_start(button_state_manager, operation_name: str, ui_logger) -> bool:
    """Check operation dengan safe error handling."""
    try:
        can_start, reason = button_state_manager.can_start_operation(operation_name)
        if not can_start:
            ui_logger.warning(f"⚠️ {reason}")
        return can_start
    except Exception as e:
        ui_logger.error(f"❌ Error cek operation state: {str(e)}")
        return False

def safe_update_status_panel(ui_components: Dict[str, Any], message: str, status: str) -> None:
    """Update status panel dengan safe error handling."""
    try:
        if 'status_panel' in ui_components:
            from smartcash.ui.components import update_status_panel
            update_status_panel(ui_components['status_panel'], message, status)
    except Exception:
        pass  # Silent fail untuk status panel

def safe_progress_update(ui_components: Dict[str, Any], value: int, message: str, progress_type: str = 'overall') -> None:
    """Safe progress update dengan multiple fallbacks."""
    try:
        if 'update_progress' in ui_components and callable(ui_components['update_progress']):
            ui_components['update_progress'](progress_type, value, message)
        elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'update'):
            ui_components['tracker'].update(progress_type, value, message)
    except Exception:
        pass  # Silent fail untuk progress update

def safe_progress_start(ui_components: Dict[str, Any], operation: str, message: str) -> None:
    """Safe progress start."""
    try:
        if 'show_for_operation' in ui_components and callable(ui_components['show_for_operation']):
            ui_components['show_for_operation'](operation)
        elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'show'):
            ui_components['tracker'].show(operation)
    except Exception:
        pass

def safe_progress_complete(ui_components: Dict[str, Any], message: str) -> None:
    """Safe progress completion."""
    try:
        if 'complete_operation' in ui_components and callable(ui_components['complete_operation']):
            ui_components['complete_operation'](message)
        elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'complete'):
            ui_components['tracker'].complete(message)
    except Exception:
        pass

def safe_progress_error(ui_components: Dict[str, Any], message: str) -> None:
    """Safe progress error."""
    try:
        if 'error_operation' in ui_components and callable(ui_components['error_operation']):
            ui_components['error_operation'](message)
        elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'error'):
            ui_components['tracker'].error(message)
    except Exception:
        pass

def safe_clear_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Safe clear confirmation area."""
    try:
        if 'confirmation_area' in ui_components:
            ui_components['confirmation_area'].clear_output()
    except Exception:
        pass

def safe_reset_ui_state(ui_components: Dict[str, Any]) -> None:
    """Safe reset UI state."""
    try:
        ui_components['stop_requested'] = False
        ui_components['augmentation_running'] = False
        safe_clear_confirmation_area(ui_components)
    except Exception:
        pass

@contextmanager
def safe_operation_context(ui_components: Dict[str, Any], operation_name: str, ui_logger):
    """
    Safe context manager untuk operations dengan comprehensive error handling.
    
    Args:
        ui_components: Dictionary komponen UI
        operation_name: Nama operasi
        ui_logger: Logger instance
    """
    button_state_manager = None
    
    try:
        # Get button state manager
        button_state_manager = get_button_state_manager_safe(ui_components, ui_logger)
        
        if button_state_manager:
            # Use proper context manager
            with button_state_manager.operation_context(operation_name):
                yield button_state_manager
        else:
            # Fallback tanpa button state management
            ui_logger.warning(f"⚠️ Fallback mode untuk {operation_name}")
            yield None
            
    except Exception as e:
        ui_logger.error(f"🔥 Error dalam operation context {operation_name}: {str(e)}")
        raise
    finally:
        # Additional cleanup jika diperlukan
        safe_reset_ui_state(ui_components)

def safe_import_and_call(module_path: str, function_name: str, *args, fallback_result=None, **kwargs):
    """
    Safe import dan call function dengan fallback.
    
    Args:
        module_path: Path ke modul
        function_name: Nama fungsi yang akan dipanggil
        *args: Arguments untuk fungsi
        fallback_result: Hasil fallback jika gagal
        **kwargs: Keyword arguments untuk fungsi
        
    Returns:
        Hasil fungsi atau fallback_result jika gagal
    """
    try:
        import importlib
        module = importlib.import_module(module_path)
        func = getattr(module, function_name)
        return func(*args, **kwargs)
    except Exception:
        return fallback_result

def create_safe_callback(ui_components: Dict[str, Any], callback_name: str) -> Callable:
    """
    Create safe callback function dengan error handling.
    
    Args:
        ui_components: Dictionary komponen UI
        callback_name: Nama callback untuk logging
        
    Returns:
        Safe callback function
    """
    def safe_callback(*args, **kwargs) -> bool:
        try:
            # Cek stop signal dengan null safety
            if ui_components.get('stop_requested', False):
                return False
            
            # Process callback logic here berdasarkan kebutuhan
            return True
            
        except Exception:
            # Silent fail untuk callback - jangan break main process
            return True
    
    return safe_callback

def validate_ui_components(ui_components: Dict[str, Any], required_keys: list = None) -> Tuple[bool, list]:
    """
    Validasi UI components dengan list required keys.
    
    Args:
        ui_components: Dictionary komponen UI
        required_keys: List key yang wajib ada
        
    Returns:
        Tuple (is_valid, missing_keys)
    """
    if not ui_components or not isinstance(ui_components, dict):
        return False, ['ui_components tidak valid']
    
    if not required_keys:
        return True, []
    
    missing_keys = [key for key in required_keys if key not in ui_components]
    return len(missing_keys) == 0, missing_keys

def safe_get_widget_value(ui_components: Dict[str, Any], widget_key: str, default_value=None):
    """
    Safe get widget value dengan fallback.
    
    Args:
        ui_components: Dictionary komponen UI
        widget_key: Key widget
        default_value: Nilai default jika gagal
        
    Returns:
        Nilai widget atau default_value
    """
    try:
        widget = ui_components.get(widget_key)
        if widget and hasattr(widget, 'value'):
            return widget.value
        return default_value
    except Exception:
        return default_value

def safe_set_widget_value(ui_components: Dict[str, Any], widget_key: str, value) -> bool:
    """
    Safe set widget value.
    
    Args:
        ui_components: Dictionary komponen UI
        widget_key: Key widget
        value: Nilai yang akan di-set
        
    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        widget = ui_components.get(widget_key)
        if widget and hasattr(widget, 'value'):
            widget.value = value
            return True
        return False
    except Exception:
        return False