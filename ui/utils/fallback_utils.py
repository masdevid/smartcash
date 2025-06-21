"""
File: smartcash/ui/utils/fallback_utils.py
Deskripsi: Centralized fallback utilities with traceback support for UI components
"""

import ipywidgets as widgets
import importlib
import traceback
import sys
from typing import Dict, Any, Optional, Callable, Tuple, Union, List
from IPython.display import display, HTML
from dataclasses import dataclass

@dataclass
class FallbackConfig:
    """Configuration for fallback UI components"""
    title: str = "⚠️ Error"
    message: str = "An error occurred"
    traceback: str = ""
    module_name: str = ""
    show_traceback: bool = True
    show_retry: bool = False
    retry_callback: Optional[Callable] = None
    container_style: Dict[str, str] = None
    
    def __post_init__(self):
        if self.container_style is None:
            self.container_style = {
                'border': '1px solid #f5c6cb',
                'border_radius': '8px',
                'padding': '10px',
                'margin': '10px 0',
                'background': '#f8d7da',
                'color': '#721c24'
            }


def import_with_fallback(module_path: str, fallback_value: Any = None) -> Any:
    """Import modul dengan fallback - one-liner style"""
    try:
        parts = module_path.split('.')
        return getattr(importlib.import_module('.'.join(parts[:-1])), parts[-1]) if '.' in module_path else importlib.import_module(module_path)
    except (ImportError, AttributeError):
        return fallback_value


class FallbackLogger:
    """Fallback logger yang menyediakan method standar logging"""
    def __init__(self, name=None):
        self.name = name or 'fallback'
        
    def debug(self, msg, *args, **kwargs):
        print(f"[DEBUG] {msg}")
        
    def info(self, msg, *args, **kwargs):
        print(f"[INFO] {msg}")
        
    def warning(self, msg, *args, **kwargs):
        print(f"[WARNING] {msg}")
        
    def error(self, msg, *args, **kwargs):
        print(f"[ERROR] {msg}")
        
    def exception(self, msg, *args, **kwargs):
        print(f"[EXCEPTION] {msg}")
        if 'exc_info' in kwargs and kwargs['exc_info']:
            import traceback
            traceback.print_exc()

def get_safe_logger(module_name: str = None) -> Any:
    """Get logger dengan fallback jika tidak tersedia"""
    logger = import_with_fallback('smartcash.common.logger.get_logger', lambda x=None: FallbackLogger(x))(module_name)
    if not hasattr(logger, 'exception'):
        # Jika logger tidak memiliki method exception, gunakan method error
        logger.exception = logger.error
    return logger


def get_safe_status_widget(ui_components: Dict[str, Any]) -> Any:
    """Get widget output status dengan fallback pattern"""
    return next((ui_components[key] for key in ['status', 'output', 'log_output'] 
                if key in ui_components and hasattr(ui_components[key], 'clear_output')), None)


def create_status_message(message: str, title: str = 'Status', status_type: str = 'info', show_icon: bool = True) -> str:
    """Create HTML status message dengan styling compatibility"""
    try:
        from smartcash.ui.utils.constants import ALERT_STYLES, ICONS
        style = ALERT_STYLES.get(status_type, ALERT_STYLES['info'])
        icon = ICONS.get(status_type, ICONS.get('info', 'ℹ️')) if show_icon else ""
    except ImportError:
        # Fallback styles
        styles = {
            'info': {'bg_color': '#d1ecf1', 'text_color': '#0c5460'},
            'success': {'bg_color': '#d4edda', 'text_color': '#155724'},
            'warning': {'bg_color': '#fff3cd', 'text_color': '#856404'},
            'error': {'bg_color': '#f8d7da', 'text_color': '#721c24'}
        }
        style = styles.get(status_type, styles['info'])
        icon = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌'}.get(status_type, 'ℹ️') if show_icon else ""
    
    return f"""
    <div style="padding:8px 12px; background-color:{style['bg_color']}; 
               color:{style['text_color']}; border-radius:4px; margin:5px 0;
               border-left:4px solid {style['text_color']};">
        <h4 style="margin:0 0 5px 0; color:{style['text_color']};">{title}</h4>
        <p style="margin:3px 0">{message}</p>
    </div>
    """


def create_fallback_ui(
    error_message: str, 
    module_name: str = "module", 
    ui_components: Dict[str, Any] = None,
    exc_info: Optional[Tuple] = None,
    config: Optional[FallbackConfig] = None
) -> Dict[str, Any]:
    """Create fallback UI dengan existing alert template compatibility"""
    logger = get_safe_logger('fallback_ui')
    
    # Initialize config with defaults if not provided
    if config is None:
        config = FallbackConfig()
    
    # Update config with provided values
    config.message = error_message or config.message
    config.module_name = module_name or config.module_name
    
    # Get traceback if not provided but exception info is available
    if exc_info and not config.traceback:
        try:
            config.traceback = ''.join(traceback.format_exception(*exc_info))
        except:
            config.traceback = "Failed to generate traceback"
    
    logger.error(f"Creating fallback UI for {config.module_name}: {config.message}")
    
    # Create error details section
    error_details = [
        f"<strong>Module:</strong> {config.module_name}",
        f"<strong>Error:</strong> {config.message}"
    ]
    
    if config.show_traceback and config.traceback:
        error_details.extend([
            "<hr style='margin: 10px 0; border: 0; border-top: 1px solid #f5c6cb;'>",
            "<strong>Details:</strong>",
            f"<pre style='background: rgba(0,0,0,0.05); padding: 10px; border-radius: 4px; overflow-x: auto;'>{config.traceback}</pre>"
        ])
    
    # Create action buttons
    buttons = []
    if config.show_retry and config.retry_callback:
        retry_button = widgets.Button(
            description="🔄 Retry",
            button_style='warning',
            layout=widgets.Layout(width='100px')
        )
        retry_button.on_click(lambda _: config.retry_callback())
        buttons.append(retry_button)
    
    # Prepare error details HTML
    error_details_html = "<br>".join(error_details)
    
    # Create the error widget
    error_widget = widgets.VBox(
        [
            widgets.HTML(
                f"""
                <div style="padding: 10px; border-radius: 4px;">
                    <h4 style="margin: 0 0 10px 0; color: #721c24;">
                        {config.title}
                    </h4>
                    <div style="margin-bottom: 10px;">
                        {error_details_html}
                    </div>
                </div>
                """
            ),
            widgets.HBox(buttons) if buttons else widgets.HTML("")
        ],
        layout=widgets.Layout(
            width='100%',
            **{k: v for k, v in config.container_style.items() if k != 'margin'}
        )
    )
    
    # Update status panel if available
    if ui_components and 'status' in ui_components:
        try:
            with ui_components['status']:
                display(HTML(
                    f"<div style='color: #721c24;'>"
                    f"<strong>Error in {config.module_name}:</strong> {config.message}"
                    f"</div>"
                ))
        except Exception as e:
            logger.error(f"Failed to update status panel: {str(e)}")
    
    return {
        'ui': error_widget,
        'error': config.message,
        'status': widgets.HTML(f"<div style='color: #721c24;'>{config.message}</div>"),
        'fallback_mode': True,
        'error_details': {
            'module': config.module_name,
            'message': config.message,
            'traceback': config.traceback if config.show_traceback else None
        }
    }


def create_error_ui(error_message: str, module_name: str = "module") -> widgets.HTML:
    """Create simple error UI widget - compatibility dengan existing usage"""
    return widgets.HTML(f"""
    <div style="padding:15px; background:#f8d7da; border:1px solid #dc3545; 
               border-radius:5px; color:#721c24; margin:10px 0;">
        <h4>⚠️ {module_name.title()} Error</h4>
        <p>{error_message}</p>
        <small>💡 Try restarting cell atau check dependencies</small>
    </div>""")


def show_status_safe(message: str, status_type: str = 'info', ui_components: Dict[str, Any] = None) -> None:
    """Show status dengan fallback ke print"""
    status_widget = get_safe_status_widget(ui_components) if ui_components else None
    
    if status_widget:
        with status_widget:
            display(HTML(create_status_message(message, status_type=status_type)))
    else:
        # Fallback print dengan emoji
        icons = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌'}
        print(f"{icons.get(status_type, 'ℹ️')} {message}")


def try_operation_safe(
    operation: Callable, 
    fallback_value: Any = None, 
    logger = None,
    operation_name: str = "operation",
    on_error: Callable[[Exception], Any] = None,
    exc_info: bool = False,
    **fallback_kwargs
) -> Any:
    """Execute operation dengan safe error handling
    
    Args:
        operation: Fungsi yang akan dijalankan
        fallback_value: Nilai yang akan dikembalikan jika terjadi error
        logger: Logger untuk mencatat log
        operation_name: Nama operasi untuk keperluan logging
        on_error: Callback yang akan dipanggil jika terjadi error
        exc_info: Flag untuk menampilkan info exception
        
    Returns:
        Hasil operasi atau fallback_value jika terjadi error
    """
    try:
        result = operation()
        if logger:
            logger.info(f"✅ {operation_name.capitalize()} berhasil")
        return result
    except Exception as e:
        error_msg = f"Gagal {operation_name}: {str(e)}"
        if logger:
            logger.error(error_msg, exc_info=True)
        
        if exc_info:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            fallback_kwargs['exc_info'] = (exc_type, exc_value, exc_traceback)
        
        if on_error:
            return on_error(e, **fallback_kwargs)
            
        return fallback_value


def update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
    """Update status panel dengan safe fallback - compatibility function"""
    if ui_components and 'status_panel' in ui_components:
        ui_components['status_panel'].value = create_status_message(message, status_type=status_type)
    else:
        show_status_safe(message, status_type, ui_components)


def create_minimal_ui(title: str = "SmartCash UI", message: str = "UI sedang dimuat...") -> Dict[str, Any]:
    """Create minimal UI untuk initial loading state"""
    status_widget = widgets.Output(layout=widgets.Layout(width='100%', min_height='100px'))
    
    ui_container = widgets.VBox([
        widgets.HTML(f"<h3>🚀 {title}</h3>"),
        widgets.HTML(f"<p>ℹ️ {message}</p>"),
        status_widget
    ], layout=widgets.Layout(width='100%', padding='15px'))
    
    return {
        'ui': ui_container,
        'status': status_widget,
        'minimal_mode': True
    }


def create_simple_container(components: list, title: str = None, container_type: str = "vbox") -> widgets.Widget:
    """Create simple container untuk components"""
    # Filter None components
    valid_components = [comp for comp in components if comp is not None]
    
    if title:
        title_widget = widgets.HTML(f"<h4 style='margin: 10px 0;'>{title}</h4>")
        valid_components.insert(0, title_widget)
    
    if container_type == "hbox":
        return widgets.HBox(valid_components, layout=widgets.Layout(width='100%'))
    else:
        return widgets.VBox(valid_components, layout=widgets.Layout(width='100%'))


def get_safe_component(ui_components: Dict[str, Any], key: str, fallback_widget=None):
    """Safely get UI component dengan fallback"""
    component = ui_components.get(key)
    if component is not None:
        return component
    
    # Create fallback widget jika tidak ada
    if fallback_widget is None:
        fallback_widget = widgets.HTML(f"<div style='color: #888;'>{key} tidak tersedia</div>")
    
    return fallback_widget


# Fungsi safe_operation yang dibutuhkan oleh evaluation module
def safe_operation(operation_func, fallback_value=None, error_handler=None):
    """Execute operation dengan safe error handling dan custom error handler"""
    try:
        return operation_func()
    except Exception as e:
        error_msg = f"Gagal {operation_func.__name__}: {str(e)}"
        if error_handler:
            try:
                return error_handler(e)
            except Exception:
                pass  # Silent fail pada error handler
        return fallback_value


def create_init_fallback_ui(
    error_msg: str,
    module_name: str,
    exc_info: Optional[Tuple] = None,
    ui_components: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Buat fallback UI yang konsisten untuk initializer
    
    Args:
        error_msg: Pesan error yang akan ditampilkan
        module_name: Nama modul untuk ditampilkan
        exc_info: Optional exception info tuple (type, value, traceback)
        ui_components: Komponen UI yang ada (opsional)
        
    Returns:
        Dictionary berisi komponen UI fallback
    """
    import ipywidgets as widgets
    from IPython.display import display
    
    # Buat widget error
    error_widget = widgets.HTML(
        f"""
        <div style="
            padding: 15px;
            margin: 10px 0;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            background-color: #f8d7da;
            color: #721c24;
        ">
            <h4 style="margin-top: 0; color: #721c24;">⚠️ Error in {module_name}</h4>
            <p style="margin-bottom: 0;">{error_msg}</p>
        </div>
        """
    )
    
    # Tampilkan traceback jika ada
    if exc_info:
        import traceback
        tb_widget = widgets.Output()
        with tb_widget:
            traceback.print_exception(*exc_info)
        
        # Gabungkan widget error dengan traceback
        container = widgets.VBox([error_widget, tb_widget])
    else:
        container = error_widget
    
    # Tampilkan widget
    display(container)
    
    return {
        'ui': container,
        'error': error_msg,
        'status': widgets.HTML(f'<div style="color: #721c24;">{error_msg}</div>'),
        'fallback_mode': True
    }

# One-liner utilities untuk common patterns
safe_import = lambda path, default=None: import_with_fallback(path, default)
safe_getattr = lambda obj, attr, default=None: getattr(obj, attr, default) if obj else default
safe_call = lambda func, *args, default=None, **kwargs: try_operation_safe(lambda: func(*args, **kwargs), default)
safe_display = lambda widget: display(widget) if widget else None
safe_update = lambda widget, attr, value: try_operation_safe(lambda: setattr(widget, attr, value), silent_fail=True)
create_info_message = lambda msg: create_status_message(msg, status_type='info')
create_success_message = lambda msg: create_status_message(msg, status_type='success')
create_error_message = lambda msg: create_status_message(msg, status_type='error')
create_warning_message = lambda msg: create_status_message(msg, status_type='warning')