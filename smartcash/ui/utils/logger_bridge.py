"""
File: smartcash/ui/utils/logger_bridge.py
Deskripsi: Fixed UI Logger Bridge untuk kompatibilitas dengan SmartCashLogger
"""

import sys
import traceback
from typing import Dict, Any, Optional, Callable

# Import logger dengan error handling
try:
    from smartcash.common.logger import get_logger, LogLevel
except ImportError:
    # Fallback logger jika tidak tersedia
    import logging
    
    class LogLevel:
        DEBUG = "DEBUG"
        INFO = "INFO"
        SUCCESS = "SUCCESS"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"
    
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
    """🔧 Fixed bridge untuk SmartCashLogger compatibility"""
    
    def __init__(self, ui_components: Dict[str, Any], logger_name: str = "ui_bridge"):
        self.ui_components = ui_components
        self.logger = get_logger(logger_name)
        self._setup_ui_callback()
    
    def debug(self, message: str) -> None:
        """Log pesan debug."""
        self._safe_log('debug', message)
    
    def info(self, message: str) -> None:
        """Log pesan info."""
        self._safe_log('info', message)
    
    def success(self, message: str) -> None:
        """Log pesan success dengan fallback."""
        if hasattr(self.logger, 'success'):
            self.logger.success(message)
        else:
            # Fallback ke info dengan prefix SUCCESS
            self._safe_log('info', f"SUCCESS: {message}")
        
        # Update UI
        self._log_to_ui('success', message)
    
    def warning(self, message: str) -> None:
        """Log pesan warning."""
        self._safe_log('warning', message)
    
    def error(self, message: str, exc_info=None) -> None:
        """Log pesan error dengan safe parameter handling."""
        # Check if logger supports exc_info parameter
        if hasattr(self.logger, 'error'):
            try:
                # Try dengan exc_info parameter
                import inspect
                sig = inspect.signature(self.logger.error)
                if 'exc_info' in sig.parameters:
                    self.logger.error(message, exc_info=exc_info)
                else:
                    # Fallback tanpa exc_info
                    self.logger.error(message)
                    if exc_info:
                        # Log exception info secara manual
                        import traceback
                        self.logger.error(f"Exception traceback: {traceback.format_exception(*exc_info)}")
            except Exception:
                # Final fallback
                self.logger.error(message)
        else:
            print(f"[ERROR] {message}")
        
        # Update UI
        self._log_to_ui('error', message)
    
    def critical(self, message: str) -> None:
        """Log pesan critical."""
        self._safe_log('critical', message)
    
    def _safe_log(self, level: str, message: str) -> None:
        """Safe logging dengan fallback handling."""
        try:
            if hasattr(self.logger, level):
                getattr(self.logger, level)(message)
            else:
                print(f"[{level.upper()}] {message}")
        except Exception as e:
            print(f"[{level.upper()}] {message}")
            print(f"[WARNING] Logger error: {str(e)}")
        
        # Update UI
        self._log_to_ui(level, message)
    
    def _log_to_ui(self, level: str, message: str) -> None:
        """Log ke UI dengan safe error handling."""
        try:
            # Try to get the log output widget
            log_output = self._get_log_output_widget()
            if not log_output:
                return
            
            # Map log levels to match the LogLevel enum in log_accordion.py
            level_mapping = {
                'debug': 'debug',
                'info': 'info',
                'success': 'success',
                'warning': 'warning',
                'error': 'error',
                'critical': 'critical'
            }
            
            # Get the mapped level, default to 'info' if not found
            mapped_level = level_mapping.get(level.lower(), 'info')
            
            # Append the log message
            log_output.append_log(
                message=message,
                level=mapped_level,
                namespace='env_config',
                module='smartcash.ui.setup.env_config'
            )
            
        except Exception as e:
            # Log the error to console if UI update fails
            print(f"[ERROR] Failed to log to UI: {str(e)}")
            print(f"[ERROR] Original message: [{level.upper()}] {message}")
    
    def _get_log_output_widget(self):
        """Get the log output widget from UI components."""
        try:
            # First try to get the log_output directly
            if 'log_output' in self.ui_components:
                log_output = self.ui_components['log_output']
                if hasattr(log_output, 'append_log'):
                    return log_output
            
            # Then try to get it from log_components if available
            if 'log_components' in self.ui_components and isinstance(self.ui_components['log_components'], dict):
                log_components = self.ui_components['log_components']
                if 'log_output' in log_components and hasattr(log_components['log_output'], 'append_log'):
                    return log_components['log_output']
            
            # Try to find any widget with 'log' in the key that has append_log method
            for key, widget in self.ui_components.items():
                if 'log' in key.lower() and hasattr(widget, 'append_log'):
                    return widget
            
            # If we get here, log a debug message
            print(f"[DEBUG] No log output widget found with append_log method. Available keys: {list(self.ui_components.keys())}")
            return None
            
        except Exception as e:
            print(f"[ERROR] Error getting log output widget: {str(e)}")
            return None
    
    def _get_output_widget(self):
        """Get output widget with multiple fallback options."""
        # Try to get the log output widget first
        log_output = self._get_log_output_widget()
        if log_output:
            return log_output
        
        # Fallback to finding any output widget with a value attribute
        for key, widget in self.ui_components.items():
            if hasattr(widget, 'value'):
                return widget
        
        # If we get here, log a debug message
        print(f"[DEBUG] No output widget found. Available keys: {list(self.ui_components.keys())}")
        return None
    
    def _setup_ui_callback(self):
        """Setup callback untuk UI integration."""
        try:
            if hasattr(self.logger, 'add_callback'):
                self.logger.add_callback(self._ui_callback)
        except Exception:
            # Silent fail if callback not supported
            pass
    
    def _ui_callback(self, level, message, **kwargs):
        """Callback untuk UI updates."""
        try:
            level_str = str(level).lower() if hasattr(level, 'name') else str(level).lower()
            self._log_to_ui(level_str, message)
        except Exception:
            # Silent fail
            pass

def create_ui_logger_bridge(ui_components: Dict[str, Any], logger_name: str = "ui_bridge") -> UILoggerBridge:
    """Factory function untuk membuat UILoggerBridge."""
    return UILoggerBridge(ui_components, logger_name)

def setup_ui_logging(ui_components: Dict[str, Any], logger_name: str = "ui_bridge") -> UILoggerBridge:
    """Setup logging untuk UI dengan safe error handling."""
    return create_ui_logger_bridge(ui_components, logger_name)