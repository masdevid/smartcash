"""
File: smartcash/ui/utils/fallback_utils.py
Deskripsi: Fixed fallback utilities untuk UI recovery dan error handling
"""

from IPython.display import display, HTML
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
import importlib
import sys
import ipywidgets as widgets

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
    """Create HTML status message dengan styling dari constants"""
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

def create_fallback_ui(error_message: str, module_name: str = 'UI', ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create fallback UI dengan existing alert template - one-liner style"""
    # Ensure ui_components valid
    ui_components = ui_components or {}
    
    # Create status widget jika belum ada
    if 'status' not in ui_components:
        ui_components['status'] = widgets.Output(layout=widgets.Layout(
            width='100%', border='1px solid #ddd', min_height='100px', padding='10px'))
    
    # Create main UI container
    if 'ui' not in ui_components:
        header = widgets.HTML(f"<h3>⚠️ {module_name} (Fallback Mode)</h3>")
        ui_components['ui'] = widgets.VBox([
            header, 
            ui_components['status']
        ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Show error message
    show_status_safe(f"🚨 {error_message}", 'error', ui_components)
    
    return ui_components

def create_error_ui(message: str, module_name: str = 'Module') -> widgets.HTML:
    """Create simple error UI widget - one-liner fallback"""
    return widgets.HTML(f"""
    <div style="padding:15px; background:#f8d7da; border:1px solid #dc3545; 
               border-radius:5px; color:#721c24; margin:10px 0;">
        <h4>⚠️ {module_name} Error</h4>
        <p>{message}</p>
        <small>💡 Try restarting cell atau check dependencies</small>
    </div>""")

def try_operation_safe(operation: Callable, fallback_value: Any = None, 
                      logger=None, operation_name: str = "operasi") -> Any:
    """Execute operation dengan safe error handling"""
    try:
        result = operation()
        if logger and result: 
            logger.info(f"✅ {operation_name.capitalize()} berhasil")
        return result
    except Exception as e:
        if logger: 
            logger.warning(f"⚠️ Error saat {operation_name}: {str(e)}")
        return fallback_value

def load_config_safe(config_path: str, logger=None) -> Dict[str, Any]:
    """Load config dengan safe fallback pattern"""
    try:
        from smartcash.common.config.manager import get_config_manager
        config_manager = get_config_manager()
        
        # Extract module name dari config_path
        module_name = (config_path.split('/')[-1].split('_')[0] 
                      if '/' in config_path else config_path).rsplit('.', 1)[0]
        
        # Try load config dengan fallback chain
        config = (config_manager.get_config(module_name) or 
                 config_manager.load_config(config_path) or {})
        
        if config and logger:
            logger.info(f"✅ Config loaded untuk {module_name}")
        
        return config
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Fallback ke empty config: {str(e)}")
        return {}

def get_service_safe(service_path: str, fallback_value: Any = None, **kwargs) -> Any:
    """Get service dengan safe instantiation"""
    ServiceClass = import_with_fallback(service_path, None)
    return try_operation_safe(lambda: ServiceClass(**kwargs), fallback_value) if ServiceClass else fallback_value

def update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
    """Update status panel dengan safe fallback - Fixed function name"""
    if ui_components and 'status_panel' in ui_components:
        ui_components['status_panel'].value = create_status_message(message, status_type=status_type)
    else:
        show_status_safe(message, status_type, ui_components)

def handle_ui_error(ui_components: Dict[str, Any], error: Exception, 
                   operation_name: str = "operasi", show_fallback: bool = True) -> Dict[str, Any]:
    """Handle UI error dengan comprehensive recovery"""
    error_message = f"Error saat {operation_name}: {str(error)}"
    
    # Log error jika logger tersedia
    logger = ui_components.get('logger') or get_safe_logger()
    if logger:
        logger.error(f"💥 {error_message}")
    
    # Show status atau create fallback UI
    if show_fallback:
        return create_fallback_ui(error_message, operation_name.title(), ui_components)
    else:
        show_status_safe(error_message, 'error', ui_components)
        return ui_components

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

# One-liner utilities untuk common patterns
safe_import = lambda path, default=None: import_with_fallback(path, default)
safe_getattr = lambda obj, attr, default=None: getattr(obj, attr, default) if obj else default
safe_call = lambda func, *args, default=None, **kwargs: try_operation_safe(lambda: func(*args, **kwargs), default)
safe_display = lambda widget: display(widget) if widget else None

# Context manager untuk safe operations
class SafeOperationContext:
    def __init__(self, ui_components: Dict[str, Any], operation_name: str = "operasi"):
        self.ui_components = ui_components
        self.operation_name = operation_name
    
    def __enter__(self):
        show_status_safe(f"🔄 Memulai {self.operation_name}...", 'info', self.ui_components)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            handle_ui_error(self.ui_components, exc_val, self.operation_name, False)
        else:
            show_status_safe(f"✅ {self.operation_name.capitalize()} selesai", 'success', self.ui_components)

def safe_operation(ui_components: Dict[str, Any], operation_name: str = "operasi"):
    """Factory untuk SafeOperationContext"""
    return SafeOperationContext(ui_components, operation_name)