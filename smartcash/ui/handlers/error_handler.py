"""
File: smartcash/ui/handlers/error_handler.py
Deskripsi: Utilitas penanganan error pada UI yang lebih sederhana dan konsisten
"""

from IPython.display import display, HTML
from typing import Dict, Any, Optional, Callable, Union, Type, TypeVar, List
from functools import wraps
import traceback
import sys

from smartcash.ui.utils.constants import ICONS, ALERT_STYLES

T = TypeVar('T')

class UIError(Exception):
    """Base class untuk error pada UI"""
    def __init__(self, message: str, ui_components: Optional[Dict[str, Any]] = None, **kwargs):
        self.message = message
        self.ui_components = ui_components
        self.extra = kwargs
        super().__init__(message)

class DependencyUIError(UIError):
    """Error khusus untuk dependency installer"""
    pass

def create_error_message(error_type: str, message: str, detail: Optional[str] = None) -> str:
    """
    Buat pesan error dengan format konsisten.
    
    Args:
        error_type: Tipe error ('error', 'warning', 'info')
        message: Pesan utama
        detail: Detail tambahan error (opsional)
        
    Returns:
        HTML string berisi pesan error
    """
    # Dapatkan style dari ALERT_STYLES
    style = ALERT_STYLES.get(error_type, ALERT_STYLES['error'])
    emoji = style['icon']
    bg_color = style['bg_color']
    text_color = style['text_color']
    
    # Buat pesan HTML
    html = f"""
    <div style="padding:10px; background-color:{bg_color}; 
               color:{text_color}; border-radius:4px; margin:5px 0;
               border-left:4px solid {text_color};">
        <p style="margin:5px 0">{emoji} {message}</p>
    """
    
    # Tambahkan detail jika ada
    if detail:
        html += f"""
        <details style="margin-top:10px;">
            <summary style="cursor:pointer;">Lihat detail error</summary>
            <pre style="margin-top:5px; white-space:pre-wrap; overflow-x:auto; 
                     font-size:0.9em; background:#f8f9fa; padding:8px; 
                     border-radius:3px;">{detail}</pre>
        </details>
        """
    
    html += "</div>"
    return html

def handle_ui_error(
    error: Exception, 
    output_widget: Any = None,
    show_traceback: bool = True,
    message: Optional[str] = None
) -> None:
    """
    Tangani error dan tampilkan pada widget output jika tersedia.
    
    Args:
        error: Exception yang terjadi
        output_widget: Widget output untuk menampilkan error
        show_traceback: Apakah perlu menampilkan traceback
        message: Pesan error kustom (jika tidak diisi, gunakan error message)
    """
    error_type = type(error).__name__
    error_msg = message or str(error)
    detail = None
    
    if show_traceback:
        detail = traceback.format_exc()
    
    error_html = create_error_message('error', f"{error_type}: {error_msg}", detail)
    
    if output_widget and hasattr(output_widget, 'clear_output'):
        with output_widget:
            display(HTML(error_html))
    else:
        # Fallback ke print jika tidak ada widget
        print(f"{ICONS['error']} {error_type}: {error_msg}")
        if show_traceback:
            print(traceback.format_exc())

def show_ui_message(
    message: str,
    message_type: str = 'info',
    output_widget: Any = None
) -> None:
    """
    Tampilkan pesan pada widget output dengan styling yang sesuai.
    
    Args:
        message: Pesan yang akan ditampilkan
        message_type: Tipe pesan ('info', 'success', 'warning', 'error')
        output_widget: Widget output untuk menampilkan pesan
    """
    message_html = create_error_message(message_type, message)
    
    if output_widget and hasattr(output_widget, 'clear_output'):
        with output_widget:
            display(HTML(message_html))
    else:
        # Fallback ke print jika tidak ada widget
        emoji = ICONS.get(message_type, ICONS['info'])
        print(f"{emoji} {message}")

def try_except_decorator(output_widget: Any = None, show_traceback: bool = True):
    """
    Decorator untuk menambahkan try-except pada fungsi.
    
    Args:
        output_widget: Widget output untuk menampilkan error
        show_traceback: Apakah perlu menampilkan traceback
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_ui_error(e, output_widget, show_traceback)
                return None
        return wrapper
    return decorator

def get_ui_component(ui_components: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Dapatkan komponen UI dengan aman.
    
    Args:
        ui_components: Dictionary komponen UI
        key: Kunci komponen yang dicari
        default: Nilai default jika tidak ditemukan
        
    Returns:
        Komponen UI atau nilai default
    """
    try:
        return ui_components.get(key, default)
    except (AttributeError, KeyError, TypeError):
        return default


def handle_ui_error(
    error_type: str = 'error',
    show_traceback: bool = True,
    log_to_ui: bool = True,
    raise_exception: bool = False,
    exception_type: Type[Exception] = UIError
):
    """
    Decorator untuk menangani error pada UI components
    
    Args:
        error_type: Tipe error ('error', 'warning', 'info')
        show_traceback: Tampilkan traceback di log
        log_to_ui: Tampilkan pesan error di UI
        raise_exception: Raise exception setelah menangani error
        exception_type: Tipe exception yang akan di-raise
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Cari parameter ui_components di args atau kwargs
            ui_components = kwargs.get('ui_components')
            if not ui_components and args and isinstance(args[0], dict):
                ui_components = args[0]
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                detail = traceback.format_exc() if show_traceback else None
                
                # Log ke console
                print(f"\n[{error_type.upper()}] {error_msg}", file=sys.stderr)
                if detail and show_traceback:
                    print(detail, file=sys.stderr)
                
                # Log ke UI jika diminta dan ui_components tersedia
                if log_to_ui and ui_components:
                    log_func = get_ui_component(ui_components, 'log_message', None)
                    if log_func and callable(log_func):
                        log_func(error_msg, error_type)
                    elif hasattr(ui_components, 'log_output'):
                        ui_components['log_output'].append_stdout(f"\n[{error_type.upper()}] {error_msg}")
                    
                    # Tampilkan detail error di UI jika tersedia
                    if detail and hasattr(ui_components, 'log_output'):
                        ui_components['log_output'].append_stdout(f"\n{detail}")
                
                # Raise exception jika diminta
                if raise_exception:
                    raise exception_type(error_msg, ui_components=ui_components) from e
                
                # Return None untuk fungsi yang mengembalikan nilai
                return None
        return wrapper
    return decorator


def safe_ui_components(ui_components_func: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """
    Decorator untuk membuat UI components dengan error handling yang lebih baik
    
    Args:
        ui_components_func: Fungsi yang mengembalikan dictionary UI components
        
    Returns:
        Fungsi wrapper yang menangani error dan mengembalikan dictionary dengan status
    """
    @wraps(ui_components_func)
    def wrapper(*args, **kwargs) -> Dict[str, Any]:
        try:
            # Coba jalankan fungsi
            result = ui_components_func(*args, **kwargs)
            
            # Pastikan hasilnya adalah dictionary
            if not isinstance(result, dict):
                raise ValueError("Fungsi harus mengembalikan dictionary")
                
            return result
            
        except Exception as e:
            # Dapatkan traceback untuk logging
            error_traceback = traceback.format_exc()
            error_type = type(e).__name__
            error_msg = str(e) or "Terjadi kesalahan yang tidak diketahui"
            
            # Buat pesan error yang informatif
            error_details = {
                'error': True,
                'module': ui_components_func.__module__ or 'unknown',
                'function': ui_components_func.__name__,
                'message': f"Gagal membuat UI components: {error_msg}",
                'error_type': error_type,
                'traceback': error_traceback,
                'args': str(args),
                'kwargs': str(kwargs)
            }
            
            # Log error ke console
            print(f"\n[ERROR] [{error_type}] {error_details['message']}", file=sys.stderr)
            print(f"Module: {error_details['module']}", file=sys.stderr)
            print(f"Function: {error_details['function']}", file=sys.stderr)
            print("\nTraceback:", file=sys.stderr)
            print(error_traceback, file=sys.stderr)
            
            # Kembalikan error details
            return error_details
            
    return wrapper


def with_ui_error_handling(ui_components_key: str = 'ui_components'):
    """
    Decorator untuk menangani error pada fungsi yang berinteraksi dengan UI components
    
    Args:
        ui_components_key: Nama parameter yang berisi UI components
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Dapatkan ui_components dari args atau kwargs
            ui_components = kwargs.get(ui_components_key)
            if not ui_components and args and isinstance(args[0], dict):
                ui_components = args[0]
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Gagal menjalankan {func.__name__}: {str(e)}"
                detail = traceback.format_exc()
                
                # Log error
                print(f"\n[ERROR] {error_msg}", file=sys.stderr)
                print(detail, file=sys.stderr)
                
                # Tampilkan error di UI jika tersedia
                if ui_components:
                    log_output = get_ui_component(ui_components, 'log_output')
                    if log_output and hasattr(log_output, 'append_stdout'):
                        log_output.append_stdout(f"\n{error_msg}")
                        log_output.append_stdout(detail)
                    
                    # Update status panel jika tersedia
                    status_panel = get_ui_component(ui_components, 'status_panel')
                    if status_panel and hasattr(status_panel, 'value'):
                        status_panel.value = create_error_message(
                            'error',
                            error_msg,
                            detail if len(detail) < 500 else detail[:500] + '...'
                        )
                
                # Re-raise exception untuk penanganan lebih lanjut jika diperlukan
                raise UIError(error_msg, ui_components=ui_components) from e
        return wrapper
    return decorator