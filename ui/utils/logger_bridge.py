"""
File: smartcash/ui/utils/logger_bridge.py
Deskripsi: Bridge untuk menghubungkan logger common dengan UI components tanpa circular dependency
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger, LogLevel


class UILoggerBridge:
    """Bridge untuk menghubungkan common logger dengan UI components tanpa circular dependency."""
    
    def __init__(self, ui_components: Dict[str, Any], namespace: str = "ui_bridge"):
        """
        Inisialisasi UI Logger Bridge.
        
        Args:
            ui_components: Dictionary komponen UI
            namespace: Namespace untuk logger
        """
        self.ui_components = ui_components
        self.namespace = namespace
        self.logger = get_logger(namespace)
        
        # Setup callback untuk UI updates
        self._setup_ui_callback()
    
    def _setup_ui_callback(self):
        """Setup callback untuk mengirim log ke UI."""
        def ui_callback(level: LogLevel, message: str):
            """Callback untuk mengirim log ke UI components."""
            try:
                # Map LogLevel ke string untuk UI
                level_map = {
                    LogLevel.DEBUG: "debug",
                    LogLevel.INFO: "info", 
                    LogLevel.SUCCESS: "success",
                    LogLevel.WARNING: "warning",
                    LogLevel.ERROR: "error",
                    LogLevel.CRITICAL: "error"
                }
                
                ui_level = level_map.get(level, "info")
                
                # Kirim ke UI jika ada komponen yang sesuai
                self._send_to_ui(message, ui_level)
                
            except Exception:
                # Jangan sampai error di callback merusak proses utama
                pass
        
        # Register callback ke logger
        self.logger.add_callback(ui_callback)
    
    def _send_to_ui(self, message: str, level: str):
        """Kirim pesan ke UI components."""
        # Import di dalam method untuk menghindari circular dependency
        try:
            from smartcash.ui.utils.ui_logger import log_to_ui
            log_to_ui(self.ui_components, message, level)
        except ImportError:
            # Fallback jika ui_logger tidak tersedia
            pass
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def success(self, message: str):
        """Log success message."""
        self.logger.success(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)


def create_ui_logger_bridge(ui_components: Dict[str, Any], namespace: str = "ui_bridge") -> UILoggerBridge:
    """
    Factory function untuk membuat UI Logger Bridge.
    
    Args:
        ui_components: Dictionary komponen UI
        namespace: Namespace untuk logger
        
    Returns:
        Instance UILoggerBridge
    """
    return UILoggerBridge(ui_components, namespace)


def log_to_service(logger, message: str, level: str = "info"):
    """
    Helper function untuk log ke service logger tanpa UI dependency.
    
    Args:
        logger: Instance logger dari common
        message: Pesan yang akan di-log
        level: Level log
    """
    if level == "success":
        logger.success(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "debug":
        logger.debug(message)
    elif level == "critical":
        logger.critical(message)
    else:
        logger.info(message)


def get_service_logger(name: str):
    """
    Helper function untuk mendapatkan logger untuk service tanpa UI dependency.
    
    Args:
        name: Nama logger
        
    Returns:
        Instance logger dari common
    """
    return get_logger(name)