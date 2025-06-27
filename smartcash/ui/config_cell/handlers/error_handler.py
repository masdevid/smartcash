"""
File: smartcash/ui/config_cell/handlers/error_handler.py
Deskripsi: Error handling untuk config cell components

Modul ini menggunakan standard error_component dari shared components.
Menerapkan fail-fast principle tanpa nested fallbacks yang berlebihan.
"""
from typing import Any, Dict, Optional, Type, TypeVar, Callable
from functools import wraps
import traceback
import logging
import ipywidgets as widgets

from smartcash.common.exceptions import SmartCashError
from smartcash.ui.components.error import create_error_component

logger = logging.getLogger(__name__)
T = TypeVar('T')

def handle_ui_errors(
    error_component_title: str = "Error",
    log_error: bool = True,
    return_type: Type[T] = dict
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """🛡️ Decorator untuk menangani error dalam method komponen UI.
    
    Args:
        error_component_title: Judul untuk komponen error
        log_error: Apakah akan log error
        return_type: Tipe return yang diharapkan dari fungsi yang didekorasi
        
    Returns:
        Decorator yang wrap fungsi dengan error handling
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                
                error_ui = create_error_response(
                    error_message=str(e),
                    error=e,
                    title=error_component_title
                )
                
                if return_type == dict:
                    return {'container': error_ui, 'error': True}
                return error_ui
                
        return wrapper
    return decorator

def create_error_response(
    error_message: str,
    error: Optional[Exception] = None,
    title: str = "Error",
    include_traceback: bool = True
) -> widgets.Widget:
    """🚨 Membuat error response menggunakan standard error_component.
    
    Args:
        error_message: Pesan error yang akan ditampilkan
        error: Exception instance opsional untuk traceback
        title: Judul untuk error component
        include_traceback: Apakah akan include traceback
        
    Returns:
        Widget error dari shared component
    """
    # Get traceback if needed
    tb_text = None
    if include_traceback and error:
        try:
            tb_text = traceback.format_exc()
            if tb_text.strip() == "NoneType: None":
                tb_text = None
        except Exception:
            tb_text = None
    
    # Use standard error component - fail fast, no nested fallbacks
    error_ui = create_error_component(
        error_message=error_message,
        title=f"🚨 {title}",
        traceback=tb_text,
        error_type="error",
        show_traceback=bool(tb_text)
    )
    
    return error_ui['container']