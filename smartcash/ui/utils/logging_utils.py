"""
File: smartcash/ui/utils/logging_utils.py
Deskripsi: Utilitas logging dengan integrasi UI yang dioptimalkan dan mekanisme thread-safe untuk menghindari masalah konkurensi
"""

import logging
import sys
import threading
from typing import Dict, Any, Optional
from datetime import datetime
from IPython.display import display, HTML

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.alert_utils import create_info_alert, create_status_indicator
from smartcash.ui.handlers.error_handler import handle_ui_error, show_ui_message

# Cache untuk handler dan logger dengan thread-safety
_logger_cache, _handler_cache = {}, {}
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
        if _in_output_context: return
            
        try:
            # Format log message dengan styling yang konsisten
            msg = self.format(record)
            
            # One-liner style mapping untuk status dan icon
            status_type = "error" if record.levelno >= logging.ERROR else "warning" if record.levelno >= logging.WARNING else "success" if record.levelno >= logging.INFO and "success" in record.msg.lower() else "info" if record.levelno >= logging.INFO else "debug"
            
            # Format dan display dengan thread-safe approach
            with _output_lock:
                _in_output_context = True
                try:
                    if hasattr(self.output_widget, 'clear_output'):
                        with self.output_widget:
                            # Gunakan komponen dari handlers yang sudah ada
                            display(create_status_indicator(status_type, msg))
                    else:
                        # Fallback dengan print jika widget tidak ada
                        icon = ICONS.get(status_type, ICONS.get("info", "ℹ️"))
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] {icon} {msg}")
                finally:
                    _in_output_context = False
            
            # Update status panel untuk level penting menggunakan komponen dari handlers
            if status_type in ['error', 'warning', 'success']:
                show_ui_message(msg, status_type, None)
                self._update_status_panel(msg, status_type)
                
            # Notify observer jika event penting
            self._notify_observer(msg, status_type, record)
            
        except Exception as e:
            print(f"Error dalam UILogHandler: {str(e)}")
            print(f"Original message: {record.msg}")
    
    def _update_status_panel(self, message: str, status_type: str):
        """Update status panel jika tersedia dengan komponen UI utils."""
        if 'status_panel' in self.ui_components:
            try:
                # Gunakan komponen dari ui_utils yang sudah ada
                self.ui_components['status_panel'].value = create_info_alert(message, status_type).value
            except Exception:
                pass
    
    def _notify_observer(self, message: str, status_type: str, record: logging.LogRecord):
        """Notify observer jika tersedia dengan integrasi observer_handler."""
        try:
            if 'observer_manager' in self.ui_components:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                
                # Event type mapping one-liner
                event_type = getattr(EventTopics, f"LOG_{status_type.upper()}", EventTopics.LOG_INFO)
                notify(event_type=event_type, sender=record.name, message=message, status=status_type, level=record.levelname, logger=record.name)
        except (ImportError, AttributeError, Exception):
            pass

class UILogger(logging.Logger):
    """Logger dengan metode tambahan untuk keperluan UI."""
    
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self._ui_components = None
    
    def set_ui_components(self, ui_components: Dict[str, Any]): self._ui_components = ui_components
    
    def success(self, msg, *args, **kwargs):
        """Log message dengan level SUCCESS."""
        if not any(icon in str(msg) for icon in ["✅", "👍", "🎉"]): msg = f"✅ {msg}"
        self.info(msg, *args, **kwargs)

    def get_ui_components(self) -> Dict[str, Any]: return self._ui_components

def setup_ipython_logging(ui_components: Dict[str, Any], module_name: str = None, log_level=logging.INFO, redirect_root: bool = False) -> Optional[UILogger]:
    """Setup logging IPython dengan integrasi UI yang lebih robust dan thread-safe."""
    global _logger_cache, _handler_cache
    
    try:
        # Validasi input dan return dummy logger jika tidak valid
        if not ui_components: return create_dummy_logger()
            
        # Normalkan nama module dan logger
        module_name = module_name or ui_components.get('module_name', 'smartcash')
        logger_name = f"smartcash.{module_name}"
        
        # Cache lookup untuk efisiensi
        if logger_name in _logger_cache:
            logger = _logger_cache[logger_name]
            if isinstance(logger, UILogger): logger.set_ui_components(ui_components)
            return logger
        
        # Setup logger dengan UILogger class
        logging.setLoggerClass(UILogger)
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        
        # Pastikan output widget tersedia dengan menggunakan handler_utils
        if 'status' not in ui_components or not ui_components['status']:
            from smartcash.ui.handlers.error_handler import get_ui_component
            from smartcash.ui.utils.layout_utils import OUTPUT_WIDGET
            
            # Create output widget dengan layout standar
            import ipywidgets as widgets
            ui_components['status'] = widgets.Output(layout=OUTPUT_WIDGET)
            
            # Tampilkan pesan untuk memastikan output widget terlihat
            with ui_components['status']:
                display(create_status_indicator('info', "🚀 Widget output untuk logging dibuat"))
        
        # Handler creation and setup with thread-safe caching
        output_widget = ui_components['status']
        handler_key = id(output_widget)
        
        with _output_lock:
            if handler_key not in _handler_cache:
                handler = UILogHandler(ui_components, level=log_level)
                handler.setFormatter(logging.Formatter('%(message)s'))
                _handler_cache[handler_key] = handler
            else:
                handler = _handler_cache[handler_key]
                handler.ui_components = ui_components
            
            # Remove handler lama dengan tipe yang sama untuk mencegah duplikasi
            for h in logger.handlers[:]:
                if isinstance(h, UILogHandler): logger.removeHandler(h)
            
            # Add handler dan kirim pesan test
            logger.addHandler(handler)
            
            # Redirect root logger jika diminta
            if redirect_root:
                root_logger = logging.getLogger()
                # Hapus semua UILogHandler untuk mencegah duplikasi
                for h in root_logger.handlers[:]:
                    if isinstance(h, UILogHandler) or isinstance(h, logging.StreamHandler):
                        root_logger.removeHandler(h)
                # Tambahkan handler UI ke root logger
                root_logger.addHandler(handler)
                # Set level root logger
                root_logger.setLevel(log_level)
        
        # Set UI components reference dan kirim test log
        if isinstance(logger, UILogger): logger.set_ui_components(ui_components)
        logger.info(f"🔄 Logger {logger_name} siap digunakan{' dengan redirect root' if redirect_root else ''}")
            
        # Cache logger untuk penggunaan berikutnya
        _logger_cache[logger_name] = logger
        
        return logger
    except Exception as e:
        # Fallback ke dummy logger dengan error handling
        print(f"Error during setup_ipython_logging: {str(e)}")
        return create_dummy_logger()

def create_dummy_logger():
    """Buat logger dummy untuk fallback dengan metode success."""
    logger = logging.getLogger('dummy_logger')
    logger.setLevel(logging.INFO)
    
    # Reset handlers untuk menghindari duplikasi
    for h in logger.handlers[:]: logger.removeHandler(h)
        
    # Tambahkan handler baru dan success method
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # Add success method dalam satu line
    if not hasattr(logger, 'success'): setattr(logger, 'success', lambda msg, *args, **kwargs: logger.info(f"✅ {msg}", *args, **kwargs))
    
    return logger

def log_to_ui(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """Log message ke UI components dengan komponen handlers/utils terintegrasi."""
    # Early exit untuk invalid input
    if not ui_components: return print(message)
    
    # Gunakan show_ui_message dari handler terintegrasi jika ada
    output_widget = ui_components.get('status')
    if output_widget:
        show_ui_message(message, status_type, output_widget)
    else:
        # Fallback ke print
        icon = ICONS.get(status_type, ICONS.get("info", "ℹ️"))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {icon} {message}")
    
    # Update status panel jika ada
    if 'status_panel' in ui_components:
        try:
            ui_components['status_panel'].value = create_info_alert(message, status_type).value
        except Exception:
            pass

def alert_to_ui(ui_components: Dict[str, Any], title: str, message: str, status_type: str = "info") -> None:
    """Tampilkan alert ke UI components dengan komponen handlers/utils terintegrasi."""
    try:
        # Gunakan komponen dari utils yang sudah ada
        content = f"<strong>{title}</strong><br>{message}"
        output_widget = ui_components.get('status')
        
        # Display dan update dengan error handling
        if output_widget and hasattr(output_widget, 'clear_output'):
            with output_widget:
                display(create_info_alert(content, status_type))
            
            # Update status panel jika tersedia
            if 'status_panel' in ui_components:
                ui_components['status_panel'].value = create_info_alert(content, status_type).value
        else:
            print(f"{title}: {message}")
    except Exception:
        print(f"{title}: {message}")

def reset_logging():
    """Reset semua konfigurasi logging ke default dengan thread safety."""
    global _logger_cache, _handler_cache, _in_output_context
    
    with _output_lock:
        # Hapus semua UILogHandler dan reset caches dalam satu loop
        for logger_name in _logger_cache:
            logger = logging.getLogger(logger_name)
            for h in logger.handlers[:]:
                if isinstance(h, UILogHandler): logger.removeHandler(h)
        
        # Juga hapus dari root logger
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            if isinstance(h, UILogHandler): root_logger.removeHandler(h)
        
        # Reset caches dan status
        _logger_cache, _handler_cache, _in_output_context = {}, {}, False
    
    # Reset default config
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                      handlers=[logging.StreamHandler(sys.stdout)])