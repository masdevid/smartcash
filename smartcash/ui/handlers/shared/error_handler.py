"""
File: smartcash/ui/handlers/shared/error_handler.py
Deskripsi: Handler untuk penanganan error di komponen UI
"""

import traceback
import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Any, Callable, Dict, Optional

from smartcash.ui.components.shared.alerts import create_status_indicator

def setup_error_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk error di UI components.
    
    Args:
        ui_components: Dictionary berisi widget UI
        
    Returns:
        Dictionary UI components yang telah ditambahkan error handler
    """
    # Tambahkan error handler jika belum ada
    if 'handle_error' not in ui_components:
        # Temukan output widget untuk error messages
        error_output = ui_components.get('status', None)
        if not error_output and 'output' in ui_components:
            error_output = ui_components['output']
            
        # Jika tidak ada output widget, gunakan widget pertama yang ditemukan
        if not error_output:
            for key, widget in ui_components.items():
                if isinstance(widget, widgets.Output):
                    error_output = widget
                    break
        
        # Tambahkan handler error ke ui_components
        ui_components['handle_error'] = create_error_handler(error_output)
        
    return ui_components

def handle_error(error: Exception, ui_output: Optional[widgets.Output] = None, clear: bool = True) -> None:
    """
    Handle error dan tampilkan di UI output.
    
    Args:
        error: Exception yang terjadi
        ui_output: Widget output untuk menampilkan error
        clear: Apakah perlu clear output sebelumnya
    """
    error_type = type(error).__name__
    error_msg = str(error)
    trace = traceback.format_exc()
    
    if ui_output:
        with ui_output:
            if clear:
                clear_output(wait=True)
            display(create_status_indicator("error", f"❌ {error_type}: {error_msg}"))
            
            # Tampilkan traceback dalam collapsed HTML
            if trace and trace != "NoneType: None":
                display(widgets.HTML(
                    f"""<details>
                        <summary style="cursor: pointer; color: #721c24;">Lihat detail error</summary>
                        <pre style="background: #f8f9fa; padding: 10px; border-radius: 4px; 
                                    color: #721c24; margin-top: 10px; font-size: 0.9em; 
                                    white-space: pre-wrap; overflow-x: auto;">{trace}</pre>
                    </details>"""
                ))
    else:
        # Fallback ke print
        print(f"❌ {error_type}: {error_msg}")

def create_error_handler(ui_output: Optional[widgets.Output] = None) -> Callable:
    """
    Buat error handler function untuk callbacks.
    
    Args:
        ui_output: Widget output untuk menampilkan error
        
    Returns:
        Function untuk handling error
    """
    def error_handler(error: Exception, clear: bool = True) -> None:
        handle_error(error, ui_output, clear)
    
    return error_handler

def try_except_decorator(ui_output: Optional[widgets.Output] = None):
    """
    Decorator untuk menambahkan try-except pada fungsi.
    
    Args:
        ui_output: Widget output untuk menampilkan error
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_error(e, ui_output)
                return None
        return wrapper
    return decorator