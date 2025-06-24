"""
File: smartcash/ui/utils/fallback_utils.py
Deskripsi: Centralized fallback utilities with traceback support for UI components
"""

import html
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
        if '.' in module_path:
            module = importlib.import_module('.'.join(parts[:-1]))
            return getattr(module, parts[-1])
        return importlib.import_module(module_path)
    except (ImportError, AttributeError):
        return fallback_value


class FallbackLogger:
    """Fallback logger yang menyediakan method standar logging"""
    def __init__(self, name=None):
        self.name = name or 'fallback'
        
    def debug(self, msg, *args, **kwargs):
        print(f"[DEBUG] {msg}")
        if kwargs.get('exc_info'):
            self._log_exception()
        
    def info(self, msg, *args, **kwargs):
        print(f"[INFO] {msg}")
        if kwargs.get('exc_info'):
            self._log_exception()
        
    def warning(self, msg, *args, **kwargs):
        print(f"[WARNING] {msg}")
        if kwargs.get('exc_info'):
            self._log_exception()
        
    def error(self, msg, *args, **kwargs):
        print(f"[ERROR] {msg}")
        if kwargs.get('exc_info'):
            self._log_exception()
    
    def exception(self, msg, *args, **kwargs):
        print(f"[EXCEPTION] {msg}")
        self._log_exception()
    
    def _log_exception(self):
        """Internal method to log exception with traceback"""
        import traceback, sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)


def get_safe_logger(module_name: str = None) -> Any:
    """Get logger dengan fallback jika tidak tersedia"""
    logger = import_with_fallback('smartcash.common.logger.get_logger', lambda x=None: FallbackLogger(x))(module_name)
    if not hasattr(logger, 'exception'):
        # Jika logger tidak memiliki method exception, gunakan method error
        logger.exception = logger.error
    return logger


def get_safe_status_widget(ui_components: Dict[str, Any]) -> Any:
    """Get widget output status dengan fallback pattern"""
    for key in ['status', 'output', 'log_output']:
        if key in ui_components and hasattr(ui_components[key], 'clear_output'):
            return ui_components[key]
    return None


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
        <h4 style="margin:0 0 5px 0; color:{style['text_color']};">{icon} {title}</h4>
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
    """Create fallback UI dengan existing alert compatibility"""
    logger = get_safe_logger('fallback_ui')
    
    try:
        # Initialize config with defaults if not provided
        if config is None:
            config = FallbackConfig()
        
        # Update config with provided values
        config.message = str(error_message) or config.message
        config.module_name = str(module_name) or config.module_name
        
        # Get traceback if not provided but exception info is available
        if exc_info and not config.traceback:
            try:
                if exc_info[0] is not None and exc_info[1] is not None:
                    config.traceback = ''.join(traceback.format_exception(*exc_info))
                else:
                    config.traceback = "No exception information available"
            except Exception as tb_error:
                config.traceback = f"Failed to generate traceback: {str(tb_error)}"
        
        logger.error(f"Creating fallback UI for {config.module_name}: {config.message}")
        if config.traceback:
            logger.error(f"Traceback:\n{config.traceback}")
        
        # Create error details section
        error_details = [
            f"<div style='margin-bottom: 10px;'><strong>Module:</strong> {html.escape(config.module_name)}</div>",
            f"<div style='margin-bottom: 10px;'><strong>Error:</strong> {html.escape(str(config.message))}</div>"
        ]
        
        if config.show_traceback and config.traceback:
            error_details.extend([
                "<hr style='margin: 10px 0; border: 0; border-top: 1px solid #f5c6cb;'>",
                "<div style='margin-bottom: 5px;'><strong>Error Details:</strong></div>",
                f"<pre style='background: rgba(0,0,0,0.05); padding: 10px; border-radius: 4px; overflow-x: auto; margin: 0; white-space: pre-wrap;'>"
                f"{html.escape(config.traceback) if isinstance(config.traceback, str) else str(config.traceback)}"
                "</pre>"
            ])
    
        # Create action buttons
        buttons = []
        if config.show_retry and config.retry_callback:
            try:
                retry_button = widgets.Button(
                    description="🔄 Retry",
                    button_style='warning',
                    layout=widgets.Layout(width='100px')
                )
                retry_button.on_click(lambda _: config.retry_callback())
                buttons.append(retry_button)
            except Exception as btn_error:
                logger.error(f"Failed to create retry button: {str(btn_error)}")
        
        # Prepare error details HTML
        error_details_html = "".join(error_details)
        
        # Create the error widget
        try:
            error_widget = widgets.VBox(
                [
                    widgets.HTML(
                        f"""
                        <div style="padding: 15px; border-radius: 4px; {config.container_style.get('border', '')};
                            background: {config.container_style.get('background', '#f8d7da')};
                            color: {config.container_style.get('color', '#721c24')};">
                            <h4 style="margin: 0 0 15px 0; padding-bottom: 8px; border-bottom: 1px solid rgba(0,0,0,0.1);">
                                {config.title}
                            </h4>
                            <div style="font-family: monospace;">
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
        except Exception as widget_error:
            logger.error(f"Failed to create error widget: {str(widget_error)}")
            # Create a minimal fallback widget
            error_widget = widgets.HTML(
                f"<div style='padding: 10px; background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; border-radius: 4px;'>"
                f"<strong>Error in {html.escape(config.module_name)}:</strong> {html.escape(str(config.message))}"
                "</div>"
            )
        
        # Update status panel if available
        status_widget = None
        if ui_components:
            try:
                if 'status' in ui_components:
                    with ui_components['status']:
                        display(HTML(
                            f"<div style='color: #721c24; font-family: monospace; white-space: pre;'>"
                            f"<strong>Error in {html.escape(config.module_name)}:</strong> {html.escape(str(config.message))}"
                            f"</div>"
                        ))
            except Exception as status_error:
                logger.error(f"Failed to update status panel: {str(status_error)}")
        
        # Create status widget for the return value
        try:
            status_widget = widgets.HTML(
                f"<div style='color: #721c24; font-family: monospace; white-space: pre;'>"
                f"{html.escape(str(config.message))}"
                "</div>"
            )
        except Exception:
            status_widget = widgets.HTML("<div>Error occurred</div>")
        
        return {
            'ui': error_widget,
            'error': str(config.message),
            'status': status_widget,
            'fallback_mode': True,
            'error_details': {
                'module': str(config.module_name),
                'message': str(config.message),
                'traceback': str(config.traceback) if config.show_traceback and config.traceback else None
            }
        }
        
    except Exception as ui_error:
        logger.error(f"Critical error in create_fallback_ui: {str(ui_error)}\n{traceback.format_exc()}")
        # Return minimal fallback UI
        return {
            'ui': widgets.HTML(f"<div style='padding:10px;color:red'>Error: {html.escape(str(ui_error))}</div>"),
            'error': str(ui_error),
            'status': widgets.HTML(f"<div>Error: {html.escape(str(ui_error))}</div>"),
            'fallback_mode': True,
            'error_details': {
                'module': 'fallback_ui',
                'message': f'Failed to create fallback UI: {str(ui_error)}',
                'traceback': traceback.format_exc()
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
            # Check if logger supports exc_info
            if hasattr(logger, 'exception'):
                logger.exception(error_msg)  # This automatically includes traceback
            else:
                # For loggers that don't support exc_info
                logger.error(error_msg)
                import traceback
                logger.error(traceback.format_exc())
        
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


# One-liner utilities untuk common patterns
safe_import = lambda path, default=None: import_with_fallback(path, default)
safe_getattr = lambda obj, attr, default=None: getattr(obj, attr, default) if obj else default
safe_call = lambda func, *args, default=None, **kwargs: try_operation_safe(lambda: func(*args, **kwargs), default)
safe_display = lambda widget: display(widget) if widget else None
create_info_message = lambda msg: create_status_message(msg, status_type='info')
create_success_message = lambda msg: create_status_message(msg, status_type='success')
create_error_message = lambda msg: create_status_message(msg, status_type='error')
create_warning_message = lambda msg: create_status_message(msg, status_type='warning')