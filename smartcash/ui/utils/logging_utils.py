"""
File: smartcash/ui/utils/logging_utils.py
Deskripsi: Utilitas logging dengan integrasi UI yang dioptimalkan dan mekanisme thread-safe untuk menghindari masalah konkurensi
"""

import logging
from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, HTML
from datetime import datetime
import sys
import threading

from smartcash.ui.utils.constants import COLORS, ICONS

# Cache untuk handler dan logger
_logger_cache = {}
_handler_cache = {}
_output_lock = threading.RLock()
_in_output_context = False

class UILogHandler(logging.Handler):
    """Handler logging yang terintegrasi dengan UI components dengan thread safety."""
    
    def __init__(self, ui_components: Dict[str, Any], level=logging.NOTSET):
        super().__init__(level)
        self.ui_components = ui_components
        self.output_widget = ui_components.get('status')
    
    def emit(self, record):
        global _in_output_context
        
        # Fallback ke print jika tidak ada output widget
        if not self.output_widget:
            print(self.format(record))
            return
            
        # Hindari rekursi infinite jika pemanggilan dalam konteks output
        if _in_output_context:
            return
            
        try:
            # Format log message dengan styling yang konsisten
            msg = self.format(record)
            
            # One-liner style mapping untuk status dan icon
            status_type = "error" if record.levelno >= logging.ERROR else \
                         "warning" if record.levelno >= logging.WARNING else \
                         "success" if record.levelno >= logging.INFO and "success" in record.msg.lower() else \
                         "info" if record.levelno >= logging.INFO else "debug"
            
            icon = ICONS.get(status_type, ICONS.get("info", "ℹ️"))
            style = f"color: {COLORS.get('danger' if status_type == 'error' else 'warning' if status_type == 'warning' else 'success' if status_type == 'success' else 'info' if status_type == 'info' else 'muted', 'gray')};"
            
            # Format dan display dengan thread-safe approach
            with _output_lock:
                _in_output_context = True
                try:
                    # Pastikan Output widget sudah diinisialisasi sebelum menggunakan with
                    if hasattr(self.output_widget, 'clear_output'):
                        with self.output_widget:
                            display(HTML(f"<span style='color: gray;'>[{datetime.now().strftime('%H:%M:%S')}]</span> <span style='{style}'>{msg}</span>"))
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] {icon} {msg}")
                finally:
                    _in_output_context = False
            
            # Update status panel untuk level penting
            if status_type in ['error', 'warning', 'success']:
                self._update_status_panel(msg, status_type)
                
            # Notify observer jika event penting
            self._notify_observer(msg, status_type, record)
            
        except Exception as e:
            print(f"Error dalam UILogHandler: {str(e)}")
            print(f"Original message: {record.msg}")
    
    def _update_status_panel(self, message: str, status_type: str):
        """Update status panel jika tersedia dengan error handling."""
        if 'status_panel' in self.ui_components:
            try:
                from smartcash.ui.utils.alert_utils import create_info_alert
                # Gunakan status panel dari ui_components, bukan fungsi
                self.ui_components['status_panel'].value = create_info_alert(message, status_type).value
            except (ImportError, Exception) as e:
                print(f"Error updating status panel: {str(e)}")
    
    def _notify_observer(self, message: str, status_type: str, record: logging.LogRecord):
        """Notify observer jika tersedia dengan error handling."""
        try:
            if 'observer_manager' in self.ui_components:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                
                # Event type mapping
                event_type_name = f"LOG_{status_type.upper()}"
                event_type = getattr(EventTopics, event_type_name, EventTopics.LOG_INFO)
                
                notify(
                    event_type=event_type,
                    sender=record.name,
                    message=message,
                    status=status_type,
                    level=record.levelname,
                    logger=record.name
                )
        except (ImportError, AttributeError, Exception):
            # Diam-diam gagal, jangan crash aplikasi
            pass

class UILogger(logging.Logger):
    """Logger dengan metode tambahan untuk keperluan UI."""
    
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self._ui_components = None
    
    def set_ui_components(self, ui_components: Dict[str, Any]):
        """Set UI components untuk logger."""
        self._ui_components = ui_components
    
    def success(self, msg, *args, **kwargs):
        """Log message dengan level SUCCESS."""
        if not any(icon in str(msg) for icon in ["✅", "👍", "🎉"]):
            msg = f"✅ {msg}"
        self.info(msg, *args, **kwargs)

    def get_ui_components(self) -> Dict[str, Any]:
        """Dapatkan UI components untuk logger ini."""
        return self._ui_components

def setup_ipython_logging(ui_components: Dict[str, Any], module_name: str = None, log_level=logging.INFO) -> Optional[UILogger]:
    """Setup logging IPython dengan integrasi UI dan error handling."""
    global _logger_cache, _handler_cache
    
    try:
        # Pastikan module_name dan logger_name
        module_name = module_name or ui_components.get('module_name', 'smartcash')
        logger_name = f"smartcash.{module_name}"
        
        # Cache check dan return untuk efisiensi
        if logger_name in _logger_cache:
            logger = _logger_cache[logger_name]
            if isinstance(logger, UILogger):
                logger.set_ui_components(ui_components)
            return logger
        
        # Setup logger dengan UILogger class
        logging.setLoggerClass(UILogger)
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        
        # Pastikan output widget tersedia dengan fallback creation
        if 'status' not in ui_components or not ui_components['status']:
            ui_components['status'] = widgets.Output(
                layout=widgets.Layout(
                    width='100%',
                    border='1px solid #ddd',
                    min_height='100px',
                    max_height='300px',
                    margin='10px 0',
                    padding='10px',
                    overflow='auto'
                )
            )
        
        # Handler creation and setup with caching
        output_widget = ui_components['status']
        handler_key = id(output_widget)
        
        if handler_key not in _handler_cache:
            handler = UILogHandler(ui_components, level=log_level)
            handler.setFormatter(logging.Formatter('%(message)s'))
            _handler_cache[handler_key] = handler
        else:
            handler = _handler_cache[handler_key]
            handler.ui_components = ui_components
        
        # Remove any existing UI handlers to avoid duplication
        for h in logger.handlers[:]:
            if isinstance(h, UILogHandler):
                logger.removeHandler(h)
        
        # Add the handler
        logger.addHandler(handler)
        
        # Set UI components and cache
        if isinstance(logger, UILogger):
            logger.set_ui_components(ui_components)
        _logger_cache[logger_name] = logger
        
        return logger
    except Exception as e:
        print(f"Error during setup_ipython_logging: {str(e)}")
        # Return a fallback logger to prevent crashes
        return create_dummy_logger()

def create_dummy_logger():
    """Buat logger dummy untuk fallback dengan metode success."""
    logger = logging.getLogger('dummy_logger')
    logger.setLevel(logging.INFO)
    
    # Reset handlers untuk menghindari duplikasi
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        
    # Tambahkan handler baru
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # Tambah metode success jika belum ada
    if not hasattr(logger, 'success'):
        setattr(logger, 'success', lambda msg, *args, **kwargs: logger.info(f"✅ {msg}", *args, **kwargs))
    
    return logger

def log_to_ui(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """Log message ke UI components dengan style yang sesuai."""
    global _in_output_context
    
    # Exit early if no output widget
    output_widget = ui_components.get('status')
    if not output_widget:
        print(message)
        return
    
    # Thread-safe styling dan icon mapping
    style = f"color: {COLORS.get('danger' if status_type == 'error' else 'warning' if status_type == 'warning' else 'success' if status_type == 'success' else 'info' if status_type == 'info' else 'muted', 'gray')};"
    icon = ICONS.get(status_type, ICONS.get("info", "ℹ️"))
    
    # Cek dan tambahkan icon jika belum ada
    icon_values = [ICONS.get(t, "") for t in ["error", "warning", "success", "info", "debug"]]
    if not any(i in message for i in icon_values if i):
        message = f"{icon} {message}"
    
    # Display dengan thread-safe context management
    with _output_lock:
        _in_output_context = True
        try:
            # Pastikan Output widget sudah diinisialisasi sebelum menggunakan with
            if hasattr(output_widget, 'clear_output'):
                with output_widget:
                    display(HTML(f"<span style='color: gray;'>[{datetime.now().strftime('%H:%M:%S')}]</span> <span style='{style}'>{message}</span>"))
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        finally:
            _in_output_context = False
    
    # Update status panel jika diperlukan
    if status_type in ['error', 'warning', 'success'] and 'status_panel' in ui_components:
        try:
            from smartcash.ui.utils.alert_utils import create_info_alert
            ui_components['status_panel'].value = create_info_alert(message, status_type).value
        except (ImportError, Exception):
            pass

def alert_to_ui(ui_components: Dict[str, Any], title: str, message: str, status_type: str = "info") -> None:
    """Tampilkan alert ke UI components dengan style yang sesuai."""
    try:
        from smartcash.ui.utils.alert_utils import create_info_alert
        output_widget = ui_components.get('status')
        content = f"<strong>{title}</strong><br>{message}"
        
        # Display dan update dengan error handling
        if output_widget and hasattr(output_widget, 'clear_output'):
            with output_widget:
                display(create_info_alert(content, status_type))
            
            # Update status panel jika tersedia
            if 'status_panel' in ui_components:
                ui_components['status_panel'].value = create_info_alert(content, status_type).value
        else:
            print(f"{title}: {message}")
    except (ImportError, Exception) as e:
        print(f"{title}: {message}")
        print(f"Error displaying alert: {str(e)}")

def reset_logging():
    """Reset semua konfigurasi logging ke default dengan thread safety."""
    global _logger_cache, _handler_cache, _in_output_context
    
    with _output_lock:
        # Hapus semua UILogHandler
        for logger_name in _logger_cache:
            logger = logging.getLogger(logger_name)
            for h in logger.handlers[:]:
                if isinstance(h, UILogHandler):
                    logger.removeHandler(h)
        
        # Reset caches
        _logger_cache = {}
        _handler_cache = {}
        _in_output_context = False
    
    # Reset default config
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )