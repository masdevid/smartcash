"""
File: smartcash/ui/utils/logging_utils.py
Deskripsi: Utilitas logging dengan integrasi UI yang dioptimalkan dan dikompres dengan one-liner style
"""

import logging
from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, HTML
from datetime import datetime
import sys

from smartcash.ui.utils.constants import COLORS, ICONS

# Cache untuk handler dan logger
_logger_cache, _handler_cache, _in_output_context = {}, {}, False

class UILogHandler(logging.Handler):
    """Handler logging yang terintegrasi dengan UI components."""
    
    def __init__(self, ui_components: Dict[str, Any], level=logging.NOTSET):
        super().__init__(level); self.ui_components = ui_components; self.output_widget = ui_components.get('status')
    
    def emit(self, record):
        global _in_output_context
        if not self.output_widget: print(self.format(record)); return
        if _in_output_context: return
            
        try:
            # Format log message with one-liner style
            msg = self.format(record)
            status_type, icon, style = ("error", ICONS.get("error", "❌"), f"color: {COLORS.get('danger', 'red')};") if record.levelno >= logging.ERROR else \
                                       ("warning", ICONS.get("warning", "⚠️"), f"color: {COLORS.get('warning', 'orange')};") if record.levelno >= logging.WARNING else \
                                       ("success", ICONS.get("success", "✅"), f"color: {COLORS.get('success', 'green')};") if record.levelno >= logging.INFO and "success" in record.msg.lower() else \
                                       ("info", ICONS.get("info", "ℹ️"), f"color: {COLORS.get('info', 'blue')};") if record.levelno >= logging.INFO else \
                                       ("debug", ICONS.get("debug", "🔍"), f"color: {COLORS.get('muted', 'gray')};")
            
            # Format and display with one-liner approach
            _in_output_context = True
            try:
                with self.output_widget: display(HTML(f"<span style='color: gray;'>[{datetime.now().strftime('%H:%M:%S')}]</span> <span style='{style}'>{msg}</span>"))
            finally: _in_output_context = False
            
            # Update status panel for important levels
            if status_type in ['error', 'warning', 'success']: self._update_status_panel(msg, status_type)
            self._notify_observer(msg, status_type, record)
        except Exception as e:
            print(f"Error dalam UILogHandler: {str(e)}"); print(f"Original message: {record.msg}")
    
    def _update_status_panel(self, message: str, status_type: str):
        """Update status panel jika tersedia."""
        if 'status_panel' in self.ui_components:
            try: 
                from smartcash.ui.utils.alert_utils import create_info_alert
                self.ui_components['status_panel'].value = create_info_alert(message, status_type).value
            except ImportError: pass
    
    def _notify_observer(self, message: str, status_type: str, record: logging.LogRecord):
        """Notify observer jika tersedia."""
        try:
            if 'observer_manager' in self.ui_components:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                
                # One-liner event type mapping
                event_type = getattr(EventTopics, f"LOG_{status_type.upper()}", EventTopics.LOG_INFO)
                
                notify(event_type=event_type, sender=record.name, message=message, 
                       status=status_type, level=record.levelname, logger=record.name)
        except (ImportError, AttributeError): pass

class UILogger(logging.Logger):
    """Logger dengan metode tambahan untuk keperluan UI."""
    
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level); self._ui_components = None
    
    def set_ui_components(self, ui_components: Dict[str, Any]):
        """Set UI components untuk logger."""
        self._ui_components = ui_components
    
    def success(self, msg, *args, **kwargs):
        """Log message dengan level SUCCESS."""
        self.info(f"✅ {msg}" if not any(icon in str(msg) for icon in ["✅", "👍", "🎉"]) else msg, *args, **kwargs)

    def get_ui_components(self) -> Dict[str, Any]:
        """Dapatkan UI components untuk logger ini."""
        return self._ui_components

def setup_ipython_logging(ui_components: Dict[str, Any], module_name: str = None, log_level=logging.INFO) -> Optional[UILogger]:
    """Setup logging IPython dengan integrasi UI."""
    global _logger_cache, _handler_cache
    
    # Pastikan module_name dan logger_name
    module_name = module_name or ui_components.get('module_name', 'smartcash')
    logger_name = f"smartcash.{module_name}"
    
    # One-liner cache check and return
    if logger_name in _logger_cache: 
        logger = _logger_cache[logger_name]
        if isinstance(logger, UILogger): logger.set_ui_components(ui_components)
        return logger
    
    # Setup logger dengan one-liner
    logging.setLoggerClass(UILogger)
    logger = logging.getLogger(logger_name); logger.setLevel(log_level)
    
    # Pastikan output widget tersedia dengan one-liner
    if 'status' not in ui_components or not ui_components['status']:
        ui_components['status'] = widgets.Output(layout=widgets.Layout(width='100%', border='1px solid #ddd', 
                                                                       min_height='100px', max_height='300px', 
                                                                       margin='10px 0', padding='10px', overflow='auto'))
    
    # Handler creation and setup with one-liner
    output_widget = ui_components['status']
    handler_key = id(output_widget)
    
    if handler_key not in _handler_cache:
        handler = UILogHandler(ui_components, level=log_level)
        handler.setFormatter(logging.Formatter('%(message)s'))
        _handler_cache[handler_key] = handler
    else:
        handler = _handler_cache[handler_key]; handler.ui_components = ui_components
    
    # Handler management with one-liner
    [logger.removeHandler(h) for h in logger.handlers[:] if isinstance(h, UILogHandler)]
    logger.addHandler(handler)
    
    # Set UI components and cache
    if isinstance(logger, UILogger): logger.set_ui_components(ui_components)
    _logger_cache[logger_name] = logger
    
    return logger

def create_dummy_logger():
    """Buat logger dummy untuk fallback."""
    logger = logging.getLogger('dummy_logger'); logger.setLevel(logging.INFO)
    [logger.removeHandler(h) for h in logger.handlers[:]]; handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s')); logger.addHandler(handler)
    if not hasattr(logger, 'success'): logger.success = lambda msg, *args, **kwargs: logger.info(f"✅ {msg}", *args, **kwargs)
    return logger

def log_to_ui(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """Log message ke UI components dengan style yang sesuai."""
    global _in_output_context
    
    # Exit early if no output widget
    output_widget = ui_components.get('status')
    if not output_widget: print(message); return
    
    # One-liner style and icon
    style = f"color: {COLORS.get('danger' if status_type == 'error' else 'warning' if status_type == 'warning' else 'success' if status_type == 'success' else 'info' if status_type == 'info' else 'muted', 'gray')};"
    icon = ICONS.get(status_type, ICONS.get("info", "ℹ️"))
    
    # One-liner icon check and add
    icon_values = [ICONS.get(t, "") for t in ["error", "warning", "success", "info", "debug"]]
    message = f"{icon} {message}" if not any(i in message for i in icon_values if i) else message
    
    # Display with one-liner context management
    _in_output_context = True
    try: 
        with output_widget: display(HTML(f"<span style='color: gray;'>[{datetime.now().strftime('%H:%M:%S')}]</span> <span style='{style}'>{message}</span>"))
    finally: _in_output_context = False
    
    # One-liner status panel update
    if status_type in ['error', 'warning', 'success'] and 'status_panel' in ui_components:
        try: 
            from smartcash.ui.utils.alert_utils import create_info_alert
            ui_components['status_panel'].value = create_info_alert(message, status_type).value
        except ImportError: pass

def alert_to_ui(ui_components: Dict[str, Any], title: str, message: str, status_type: str = "info") -> None:
    """Tampilkan alert ke UI components dengan style yang sesuai."""
    try:
        from smartcash.ui.utils.alert_utils import create_info_alert
        output_widget = ui_components.get('status')
        content = f"<strong>{title}</strong><br>{message}"
        
        # One-liner display and update
        if output_widget:
            with output_widget: display(create_info_alert(content, status_type))
            if 'status_panel' in ui_components: ui_components['status_panel'].value = create_info_alert(content, status_type).value
        else: print(f"{title}: {message}")
    except ImportError: print(f"{title}: {message}")

def reset_logging():
    """Reset semua konfigurasi logging ke default."""
    global _logger_cache, _handler_cache, _in_output_context
    
    # Reset with one-liner
    [logger.removeHandler(h) for logger_name in _logger_cache for h in logging.getLogger(logger_name).handlers[:] if isinstance(h, UILogHandler)]
    _logger_cache, _handler_cache, _in_output_context = {}, {}, False
    
    # Reset default config
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                       handlers=[logging.StreamHandler(sys.stdout)])