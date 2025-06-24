# smartcash/ui/utils/logger_bridge.py
"""
UI Logger Bridge untuk integrasi logging dengan komponen UI
Menyediakan bridge antara sistem logging dan UI dengan callback yang aman
"""

import sys
import traceback
from typing import Dict, Any, Optional, Callable, List, Union

# Import logger dengan error handling
try:
    from smartcash.common.logger import get_logger, LogLevel
except ImportError as e:
    # Fallback jika logger tidak tersedia
    import logging
    
    # Buat LogLevel dummy
    class LogLevel:
        DEBUG = "DEBUG"
        INFO = "INFO"
        SUCCESS = "SUCCESS"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"
    
    # Buat get_logger dummy
    def get_logger(name=None, level=LogLevel.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


class UILoggerBridge:
    """
    Bridge untuk menghubungkan logger dengan UI components
    Menangani callback dengan signature yang fleksibel dan delegasi logger methods
    """
    
    def __init__(self, ui_components: Dict[str, Any], logger_name: str = "ui_bridge"):
        """
        Inisialisasi UI Logger Bridge
        
        Args:
            ui_components: Dictionary komponen UI
            logger_name: Nama logger untuk namespace
        """
        self.ui_components = ui_components
        self.logger = get_logger(logger_name)
        self._callback_registered = False
        self._setup_ui_callback()
    
    # Delegate logger methods untuk kompatibilitas
    def debug(self, message: str) -> None:
        """Log pesan debug."""
        return self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log pesan info."""
        return self.logger.info(message)
    
    def success(self, message: str) -> None:
        """Log pesan success."""
        return self.logger.success(message)
    
    def warning(self, message: str) -> None:
        """Log pesan warning."""
        return self.logger.warning(message)
    
    def error(self, message: str, exc_info=None) -> None:
        """Log pesan error."""
        return self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str) -> None:
        """Log pesan critical."""
        return self.logger.critical(message)
    
    def _setup_ui_callback(self):
        """Setup callback untuk update UI."""
        if not self._callback_registered and hasattr(self.ui_components, 'update_log'):
            self.logger.add_callback(self._ui_log_callback)
            self._callback_registered = True
            
    def _ui_log_callback(self, level: LogLevel, message: str, exc_info=None):
        """
        Callback untuk mengupdate UI dengan pesan log.
        
        Args:
            level: Level log
            message: Pesan log
            exc_info: Informasi exception (opsional)
        """
        try:
            if hasattr(self.ui_components, 'update_log'):
                # Format pesan untuk UI
                formatted_msg = message
                if exc_info:
                    exc_type, exc_value, exc_traceback = exc_info
                    formatted_msg += f"\n{''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))}"
                
                # Panggil metode update_log di UI
                self.ui_components.update_log({
                    'level': level.name if hasattr(level, 'name') else str(level),
                    'message': formatted_msg,
                    'timestamp': str(datetime.datetime.now())
                })
        except Exception as e:
            # Jika terjadi error saat mengupdate UI, log ke console
            print(f"Error updating UI log: {str(e)}", file=sys.stderr)
    
    def _update_ui_status(self, level: str, message: str) -> None:
        """Update status panel UI dengan pesan log"""
        try:
            status_panel = self.ui_components.get('status_panel')
            if status_panel:
                # Format pesan dengan emoji berdasarkan level
                emoji_map = {
                    'debug': '🔍',
                    'info': 'ℹ️',
                    'success': '✅',
                    'warning': '⚠️',
                    'error': '❌',
                    'critical': '🚨'
                }
                
                emoji = emoji_map.get(level, 'ℹ️')
                formatted_message = f"{emoji} {message}"
                
                # Update dengan HTML formatting
                status_panel.value = f"<div style='color: {self._get_color_for_level(level)}'>{formatted_message}</div>"
                
        except Exception:
            pass  # Silent fail untuk update UI
    
    def _log_to_ui_output(self, level: str, message: str) -> None:
        """Log ke UI output widget jika tersedia"""
        try:
            # Coba dapatkan output widget dari berbagai kemungkinan key
            output_widget = (
                self.ui_components.get('output') or 
                self.ui_components.get('log_output') or
                (self.ui_components.get('log_accordion').children[0] 
                 if self.ui_components.get('log_accordion') else None)
            )
            
            if not output_widget:
                return
                
            # Format pesan dengan timestamp dan styling
            import datetime
            from IPython.display import display
            import ipywidgets as widgets
            
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            color = self._get_color_for_level(level)
            
            # Buat HTML widget untuk log entry
            log_entry = widgets.HTML(
                value=f"""
                <div style='margin: 2px 0; padding: 4px; word-wrap: break-word; 
                            overflow-wrap: break-word; white-space: pre-wrap; max-width: 100%; 
                            overflow: hidden;'>
                    <span style='color: #666; font-size: 12px;'>{timestamp}</span> 
                    <span style='color: {color}; font-weight: bold;'>{level.upper()}</span>
                    <span style='color: #333;'>{message}</span>
                </div>
                """
            )
            
            # Tampilkan log entry di output widget
            with output_widget:
                display(log_entry)
                
            # Auto-scroll ke bawah jika memungkinkan
            if hasattr(output_widget, 'scroll_to_bottom'):
                output_widget.scroll_to_bottom()
                    
        except Exception as e:
            # Jangan gagal hanya karena logging
            import sys
            sys.stderr.write(f"Log output error: {str(e)}\n")
            sys.stderr.write(f"[ORIGINAL {level.upper()}] {message}\n")
    
    def _get_color_for_level(self, level: str) -> str:
        """Get warna CSS untuk level log"""
        color_map = {
            'debug': '#6c757d',
            'info': '#0dcaf0',
            'success': '#198754',
            'warning': '#ffc107',
            'error': '#dc3545',
            'critical': '#dc3545'
        }
        return color_map.get(level, '#0dcaf0')
    
    def cleanup(self) -> None:
        """Cleanup callback registration"""
        if self._callback_registered:
            # Note: Logger belum mendukung remove_callback by instance
            # Untuk sekarang, marking sebagai inactive
            self._callback_registered = False
    
    def update_ui_components(self, new_ui_components: Dict[str, Any]) -> None:
        """Update UI components reference"""
        self.ui_components.update(new_ui_components)
    
    def log_to_ui(self, message: str, level: str = 'info', icon: str = None) -> None:
        """
        Direct logging ke UI tanpa melalui logger system
        Untuk situasi yang memerlukan kontrol langsung
        """
        if icon:
            message = f"{icon} {message}"
        
        # Konversi string level ke LogLevel enum jika perlu
        level_enum = getattr(LogLevel, level.upper(), LogLevel.INFO)
        
        # Direct update UI
        self._update_ui_status(level, message)
        self._log_to_ui_output(level, message)


# Factory functions untuk kemudahan penggunaan
def create_ui_logger_bridge(ui_components: Dict[str, Any], logger_name: str = "ui_bridge") -> 'UILoggerBridge':
    """
    Factory function untuk membuat UILoggerBridge
    
    Args:
        ui_components: Dictionary berisi komponen UI
        logger_name: Nama logger
        
    Returns:
        Instance UILoggerBridge
    """
    return UILoggerBridge(ui_components, logger_name)


def setup_ui_logging(ui_components: Dict[str, Any], logger_name: str = "ui_bridge") -> 'UILoggerBridge':
    """
    Setup logging untuk UI dengan satu fungsi
    
    Args:
        ui_components: Dictionary berisi komponen UI
        logger_name: Nama logger
        
    Returns:
        Instance UILoggerBridge untuk kontrol lebih lanjut
    """
    bridge = create_ui_logger_bridge(ui_components, logger_name)
    return bridge


def log_to_ui_safe(ui_components: Dict[str, Any], message: str, level: str = 'info', icon: str = None) -> None:
    """
    Helper function untuk logging ke UI dengan error handling
    Fallback ke print jika UI tidak tersedia
    
    Args:
        ui_components: Dictionary berisi komponen UI
        message: Pesan yang akan dicatat
        level: Level log (debug, info, warning, error, critical)
        icon: Icon opsional untuk ditampilkan di UI
    """
    try:
        if hasattr(ui_components, 'log_message'):
            ui_components.log_message(message, level, icon)
        else:
            print(f"[{level.upper()}] {message}")
    except Exception as e:
        print(f"[ERROR] Failed to log to UI: {str(e)}", file=sys.stderr)
        print(f"[ORIGINAL {level.upper()}] {message}", file=sys.stderr)


# Backward compatibility functions
def show_status_safe(message: str, status_type: str, ui_components: Dict[str, Any]) -> None:
    """
    Backward compatibility untuk show_status_safe
    
    Args:
        message: Pesan yang akan ditampilkan
        status_type: Tipe status (info, warning, error, dll.)
        ui_components: Dictionary berisi komponen UI
    """
    log_to_ui_safe(ui_components, message, status_type)


def update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
    """
    Backward compatibility untuk update_status_panel
    
    Args:
        ui_components: Dictionary berisi komponen UI
        message: Pesan yang akan ditampilkan
        status_type: Tipe status (info, warning, error, dll.)
    """
    log_to_ui_safe(ui_components, message, status_type)