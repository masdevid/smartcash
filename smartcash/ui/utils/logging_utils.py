"""
File: smartcash/ui/utils/logging_utils.py
Deskripsi: Utilitas logging dengan integrasi UI yang dioptimalkan dan thread-safe untuk menghindari masalah konkurensi
"""

import logging
import sys
import io
import threading
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, HTML

from smartcash.ui.utils.constants import COLORS, ICONS, ALERT_STYLES

# Simpan referensi global untuk memudahkan reset
_root_handlers = []
_registered_loggers = set()
_handler_cache = {}
_output_lock = threading.RLock()

class UILogHandler(logging.Handler):
    """Handler logging yang terintegrasi dengan UI components."""
    
    def __init__(self, ui_components: Dict[str, Any], level=logging.NOTSET, handler_name: str = None):
        """
        Inisialisasi UILogHandler.
        
        Args:
            ui_components: Dictionary berisi komponen UI
            level: Level minimum logging yang akan ditampilkan
            handler_name: Nama unik handler, berguna untuk membedakan handler observer
        """
        super().__init__(level)
        self.ui_components = ui_components
        self.output_widget = ui_components.get('status')
        self._emit_depth = 0  # Untuk mencegah rekursi
        self.name = handler_name or "ui_handler"
    
    def emit(self, record):
        """Tampilkan log di widget output dengan styling alert."""
        # Cegah rekursi
        if self._emit_depth > 0:
            return
            
        # Cek apakah output widget tersedia
        if not self.output_widget:
            print(self.format(record))
            return
            
        try:
            self._emit_depth += 1  # Tandai sedang dalam emit
            
            # Format log message
            msg = self.format(record)
            
            # Maps logging levels to alert styles
            level_to_style = {
                logging.DEBUG: 'info',
                logging.INFO: 'info',
                logging.WARNING: 'warning', 
                logging.ERROR: 'error',
                logging.CRITICAL: 'error',
            }
            
            # Ignore logs dari beberapa modul yang terlalu verbose
            ignored_modules = ['matplotlib', 'PIL', 'IPython', 'ipykernel', 'traitlets']
            if record.name and any(record.name.startswith(mod) for mod in ignored_modules):
                return
                
            # Tambahkan icon success jika memiliki kata success
            if 'success' in msg.lower() and record.levelno == logging.INFO:
                if not any(emoji in msg for emoji in ['✅', '✓', '🎉']):
                    msg = f"✅ {msg}"
                    
            # Tentukan style berdasarkan level
            style = level_to_style.get(record.levelno, 'info')
            alert_style = ALERT_STYLES.get(style, ALERT_STYLES['info'])
            icon = alert_style.get('icon', ICONS.get(style, ICONS.get('info', 'ℹ️')))
            
            # Format log message dengan styling alert
            log_html = f"""
            <div style="padding: 5px; margin: 2px 0; 
                      background-color: {alert_style['bg_color']}; 
                      color: {alert_style['text_color']}; 
                      border-left: 4px solid {alert_style['border_color']}; 
                      border-radius: 4px;">
                <span style="margin-right: 8px;">{icon}</span>
                <span>{msg}</span>
            </div>
            """
            
            # Gunakan with-lock untuk memastikan thread-safety
            with _output_lock:
                with self.output_widget:
                    display(HTML(log_html))
            
            # Update status panel untuk level penting
            if record.levelno >= logging.WARNING or ('success' in msg.lower() and record.levelno == logging.INFO):
                self._update_status_panel(msg, style)
                
            # Notify observer
            self._notify_observer(msg, style, record)
                
        except Exception as e:
            # Fallback ke default behavior
            print(f"Error in UILogHandler: {str(e)}")
            print(f"Original message: {record.msg}")
        finally:
            self._emit_depth -= 1  # Reset flag
    
    def _update_status_panel(self, message: str, status_type: str):
        """Update status panel jika tersedia."""
        if 'status_panel' in self.ui_components:
            try:
                from smartcash.ui.utils.alert_utils import create_info_alert
                self.ui_components['status_panel'].value = create_info_alert(message, status_type).value
            except Exception:
                pass
    
    def _notify_observer(self, message: str, status_type: str, record: logging.LogRecord):
        """Notify observer jika tersedia."""
        try:
            if 'observer_manager' in self.ui_components:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                
                # Map status type ke event type
                status_map = {
                    'info': EventTopics.LOG_INFO,
                    'warning': EventTopics.LOG_WARNING,
                    'error': EventTopics.LOG_ERROR,
                    'success': EventTopics.LOG_INFO
                }
                
                event_type = status_map.get(status_type, EventTopics.LOG_INFO)
                notify(event_type=event_type, sender=record.name, message=message, level=record.levelname)
        except (ImportError, AttributeError, Exception):
            pass

class UILogger(logging.Logger):
    """Logger dengan metode tambahan untuk UI."""
    
    def success(self, msg, *args, **kwargs):
        """Log message dengan level INFO tapi dengan style success."""
        # Tambahkan emoji success jika belum ada
        if not any(emoji in str(msg) for emoji in ['✅', '✓', '🎉']):
            msg = f"✅ {msg}"
        self.info(msg, *args, **kwargs)

def setup_ipython_logging(
    ui_components: Dict[str, Any],
    module_name: str = 'ui_logger',
    log_level: int = logging.INFO,
    redirect_root: bool = True,
    keep_observers: bool = True
) -> Optional[logging.Logger]:
    """
    Setup logging untuk notebook dengan integrasi UI.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        module_name: Nama modul untuk logger
        log_level: Level logging minimum
        redirect_root: Redirect root logger juga
        
    Returns:
        Logger instance atau None jika gagal
    """
    global _registered_loggers, _root_handlers
    
    try:
        # Validasi ui_components
        if not ui_components:
            return create_dummy_logger()
            
        # Dapatkan output widget
        output_widget = None
        for key in ['status', 'log_output', 'output']:
            if key in ui_components and isinstance(ui_components[key], widgets.Output):
                output_widget = ui_components[key]
                break
        
        if not output_widget:
            output_widget = widgets.Output(
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
            ui_components['status'] = output_widget
        
        # Set UILogger sebagai logger class
        original_logger_class = logging.getLoggerClass()
        logging.setLoggerClass(UILogger)
        
        # Create logger untuk modul
        logger_name = f"smartcash.{module_name}" if module_name else "smartcash"
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        
        # Kembalikan ke logger class original
        logging.setLoggerClass(original_logger_class)
        
        # Simpan di registered loggers
        _registered_loggers.add(logger_name)
        
        # Hapus UI handlers yang mungkin sudah ada kecuali observer handlers
        for handler in logger.handlers[:]:
            if isinstance(handler, UILogHandler) and (not keep_observers or 'observer' not in str(handler)):
                logger.removeHandler(handler)
        
        # Buat dan tambahkan UI handler
        ui_handler = UILogHandler(ui_components, level=log_level, handler_name=f"{module_name}_handler")
        ui_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ui_handler)
        
        # Nonaktifkan propagasi ke parent loggers
        logger.propagate = False
        
        # Redirect root logger jika diminta
        if redirect_root:
            root_logger = logging.getLogger()
            
            # Simpan handler lama dan hapus
            for handler in root_logger.handlers[:]:
                if isinstance(handler, (logging.StreamHandler, UILogHandler)):
                    root_logger.removeHandler(handler)
            
            # Tambahkan UI handler ke root logger
            root_handler = UILogHandler(ui_components, level=log_level, handler_name="root_logger_handler")
            root_handler.setFormatter(logging.Formatter('%(message)s'))
            root_logger.addHandler(root_handler)
            
            # Track handler untuk reset
            _root_handlers.append(root_handler)
            
            # Set level ke DEBUG agar semua log tertangkap
            root_logger.setLevel(logging.DEBUG)
        
        # Kirim test log
        logger.info(f"🔄 Logger {logger_name} siap digunakan")
        
        return logger
        
    except Exception as e:
        print(f"Error saat setup logger: {str(e)}")
        return create_dummy_logger()

def create_dummy_logger() -> logging.Logger:
    """
    Buat dummy logger yang aman digunakan.
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger('dummy_logger')
    
    # Reset handlers
    logger.handlers = []
    
    # Tambahkan handler dan formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # Nonaktifkan propagasi
    logger.propagate = False
    
    # Tambahkan method success
    if not hasattr(logger, 'success'):
        setattr(logger, 'success', lambda msg, *args, **kwargs: logger.info(f"✅ {msg}", *args, **kwargs))
    
    return logger

def log_to_ui(ui_components: Dict[str, Any], message: str, status: str = 'info') -> None:
    """
    Log message langsung ke UI output.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        message: Pesan yang akan ditampilkan
        status: Status pesan ('info', 'warning', 'error', 'success')
    """
    # Dapatkan output widget
    output_widget = None
    for key in ['status', 'log_output', 'output']:
        if key in ui_components and isinstance(ui_components[key], widgets.Output):
            output_widget = ui_components[key]
            break
    
    # Fallback ke print jika tidak ada output widget
    if not output_widget:
        print(f"[{status.upper()}] {message}")
        return
    
    # Get style dan icon
    alert_style = ALERT_STYLES.get(status, ALERT_STYLES['info'])
    icon = alert_style.get('icon', ICONS.get(status, ICONS.get('info', 'ℹ️')))
    
    # Format log message dengan styling alert
    log_html = f"""
    <div style="padding: 5px; margin: 2px 0; 
              background-color: {alert_style['bg_color']}; 
              color: {alert_style['text_color']}; 
              border-left: 4px solid {alert_style['border_color']}; 
              border-radius: 4px;">
        <span style="margin-right: 8px;">{icon}</span>
        <span>{message}</span>
    </div>
    """
    
    # Display di output widget
    with output_widget:
        display(HTML(log_html))
    
    # Update status panel jika ada
    if 'status_panel' in ui_components:
        try:
            from smartcash.ui.utils.alert_utils import create_info_alert
            ui_components['status_panel'].value = create_info_alert(message, status).value
        except Exception:
            pass

def alert_to_ui(ui_components: Dict[str, Any], message: str, status: str = 'info') -> None:
    """
    Tampilkan alert di UI output.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        message: Pesan yang akan ditampilkan
        status: Status pesan ('info', 'warning', 'error', 'success')
    """
    # Dapatkan output widget
    output_widget = None
    for key in ['status', 'log_output', 'output']:
        if key in ui_components and isinstance(ui_components[key], widgets.Output):
            output_widget = ui_components[key]
            break
    
    # Fallback ke print jika tidak ada output widget
    if not output_widget:
        print(f"[{status.upper()}] {message}")
        return
    
    try:
        from smartcash.ui.utils.alert_utils import create_info_alert
        with output_widget:
            display(create_info_alert(message, status))
    except ImportError:
        # Fallback jika tidak bisa import alert_utils
        log_to_ui(ui_components, message, status)

def reset_logging():
    """Reset semua konfigurasi logging ke default."""
    global _registered_loggers, _root_handlers, _handler_cache
    
    try:
        # Reset root logger
        root_logger = logging.getLogger()
        
        # Hapus UI handlers dari root logger tapi simpan referensinya
        removed_handlers = []
        for handler in list(_root_handlers):
            if handler in root_logger.handlers:
                root_logger.removeHandler(handler)
                removed_handlers.append(handler)
        
        # Reset registered loggers tapi jangan hapus observer handlers
        for logger_name in _registered_loggers:
            logger = logging.getLogger(logger_name)
            # Hapus semua UILogHandlers kecuali observer handler
            for handler in logger.handlers[:]:
                if isinstance(handler, UILogHandler) and 'observer' not in str(handler):
                    logger.removeHandler(handler)
            # Biarkan propagasi non-aktif untuk mencegah duplikasi log
            # logger.propagate = False
        
        # Reset tracking variables secara parsial
        _root_handlers = [h for h in _root_handlers if h not in removed_handlers]
        # Jangan reset _registered_loggers agar observer tetap terdaftar
        
        # Kembalikan root logger ke konfigurasi standar
        for handler in root_logger.handlers[:]:
            if isinstance(handler, UILogHandler) and 'observer' not in str(handler):
                root_logger.removeHandler(handler)
                
        # Tambahkan StreamHandler standar jika tidak ada
        has_stream_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, UILogHandler) 
                               for h in root_logger.handlers)
        if not has_stream_handler:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
            root_logger.addHandler(handler)
            
        # Reset level ke INFO tapi jangan terlalu rendah
        root_logger.setLevel(logging.INFO)
            
    except Exception as e:
        print(f"Error saat reset logging: {str(e)}")