"""
File: smartcash/ui/handlers/error_handler.py
Deskripsi: Utilitas penanganan error pada UI yang lebih sederhana dan konsisten
"""

from IPython.display import display, HTML
from typing import Dict, Any, Optional, Callable, Union
import traceback
from smartcash.ui.utils.constants import ICONS, ALERT_STYLES

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
    return ui_components.get(key, default)