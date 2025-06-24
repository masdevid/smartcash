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
            # Cari output widget
            output_widget = self._get_output_widget()
            if not output_widget:
                # Debug: Print available keys for troubleshooting
                available_keys = [k for k in self.ui_components.keys() if 'log' in k.lower() or 'output' in k.lower()]
                print(f"[DEBUG] No output widget found. Available log-related keys: {available_keys}")
                return
            
            # Format message
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            # Color mapping
            colors = {
                'debug': '#6c757d',
                'info': '#007bff', 
                'success': '#28a745',
                'warning': '#ffc107',
                'error': '#dc3545',
                'critical': '#dc3545'
            }
            
            # Emoji mapping
            emojis = {
                'debug': '🔍',
                'info': 'ℹ️',
                'success': '✅', 
                'warning': '⚠️',
                'error': '❌',
                'critical': '🔥'
            }
            
            color = colors.get(level, '#333333')
            emoji = emojis.get(level, '📝')
            
            # Format HTML message
            formatted_msg = f"""
            <div style='margin: 2px 0; padding: 4px; word-wrap: break-word; 
                        overflow-wrap: break-word; white-space: pre-wrap; max-width: 100%; 
                        overflow: hidden;'>
                <span style='color: #666; font-size: 12px;'>{timestamp}</span> 
                <span style='color: {color}; font-weight: bold;'>{level.upper()}</span>
                <span>{emoji} {message}</span>
            </div>
            """
            
            # Append to output with error handling
            try:
                current_content = getattr(output_widget, 'value', '')
                # Limit the size of the log to prevent performance issues
                max_length = 50000  # Keep last ~50KB of logs
                if len(current_content) > max_length:
                    current_content = current_content[-max_length:]
                output_widget.value = current_content + formatted_msg
                
                # Ensure the widget is visible in the UI
                if hasattr(output_widget, 'scroll_to_bottom'):
                    output_widget.scroll_to_bottom()
                
            except Exception as e:
                print(f"[ERROR] Failed to update log widget: {str(e)}")
                # Try to log the error to console as fallback
                print(f"[{timestamp}] [{level.upper()}] {message}")
            
        except Exception as e:
            # Log the error to console if UI update fails
            print(f"[ERROR] Failed to log to UI: {str(e)}")
            print(f"[ERROR] Original message: [{level.upper()}] {message}")
    
    def _get_output_widget(self):
        """Get output widget dengan multiple fallback options."""
        # Try various possible keys for the log widget
        for key in ['log_output', 'output', 'log_widget', 'log_accordion']:
            if key in self.ui_components:
                widget = self.ui_components[key]
                # If it's an accordion, try to get its output widget
                if key == 'log_accordion' and hasattr(widget, 'children') and widget.children:
                    # Look for an Output widget in the accordion's children
                    for child in widget.children:
                        if hasattr(child, 'value'):
                            return child
                # If it's a direct output widget
                elif hasattr(widget, 'value'):
                    return widget
        
        # Try to find any widget with 'log' in the key that might be an output widget
        for key, widget in self.ui_components.items():
            if 'log' in key.lower() and hasattr(widget, 'value'):
                return widget
                
        # Last resort: check if there's a log_components dictionary with an output widget
        if 'log_components' in self.ui_components and isinstance(self.ui_components['log_components'], dict):
            for key, widget in self.ui_components['log_components'].items():
                if hasattr(widget, 'value'):
                    return widget
        
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