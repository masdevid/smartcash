"""
File: smartcash/ui/utils/fallback_utils.py
Deskripsi: Merged fallback utilities dengan compatibility untuk existing imports
"""

import ipywidgets as widgets
import importlib
from typing import Dict, Any, Optional, Callable
from IPython.display import display, HTML


def import_with_fallback(module_path: str, fallback_value: Any = None) -> Any:
    """Import modul dengan fallback - one-liner style"""
    try:
        parts = module_path.split('.')
        return getattr(importlib.import_module('.'.join(parts[:-1])), parts[-1]) if '.' in module_path else importlib.import_module(module_path)
    except (ImportError, AttributeError):
        return fallback_value


def get_safe_logger(module_name: str = None) -> Any:
    """Get logger dengan fallback jika tidak tersedia"""
    return import_with_fallback('smartcash.common.logger.get_logger', lambda x=None: None)(module_name)


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


def create_fallback_ui(error_message: str, module_name: str = "module", ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create fallback UI dengan existing alert template compatibility"""
    ui_components = ui_components or {}
    
    # Create status widget jika belum ada
    if 'status' not in ui_components:
        ui_components['status'] = widgets.Output(layout=widgets.Layout(
            width='100%', border='1px solid #ddd', min_height='100px', padding='10px'))
    
    # Create main UI container
    if 'ui' not in ui_components:
        header = widgets.HTML(f"<h3>⚠️ {module_name.title()} (Fallback Mode)</h3>")
        ui_components['ui'] = widgets.VBox([
            header, 
            ui_components['status']
        ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Show error message
    show_status_safe(f"🚨 {error_message}", 'error', ui_components)
    
    # Compatibility keys
    ui_components.update({
        'main_container': ui_components['ui'],
        'error_widget': ui_components['status'],
        'error': error_message
    })
    
    return ui_components


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


def try_operation_safe(operation: Callable, fallback_value: Any = None, 
                      logger=None, operation_name: str = "operasi",
                      on_error: Callable[[Exception], Any] = None) -> Any:
    """Execute operation dengan safe error handling
    
    Args:
        operation: Fungsi yang akan dijalankan
        fallback_value: Nilai yang akan dikembalikan jika terjadi error
        logger: Logger untuk mencatat log
        operation_name: Nama operasi untuk keperluan logging
        on_error: Callback yang akan dipanggil jika terjadi error
        
    Returns:
        Hasil operasi atau fallback_value jika terjadi error
    """
    try:
        result = operation()
        if logger and result: 
            logger.info(f"✅ {operation_name.capitalize()} berhasil")
        return result
    except Exception as e:
        if logger: 
            logger.warning(f"⚠️ Error saat {operation_name}: {str(e)}")
        if callable(on_error):
            try:
                return on_error(e)
            except Exception as callback_error:
                if logger:
                    logger.error(f"⚠️ Error dalam callback on_error: {str(callback_error)}")
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
        if error_handler:
            try:
                error_handler(e)
            except Exception:
                pass  # Silent fail pada error handler
        return fallback_value


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