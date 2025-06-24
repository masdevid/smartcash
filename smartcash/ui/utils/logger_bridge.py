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
    """🔧 Fixed bridge untuk SmartCashLogger compatibility with buffering support"""
    
    def __init__(self, ui_components: Dict[str, Any], logger_name: str = "ui_bridge"):
        self.ui_components = ui_components
        self.logger = get_logger(logger_name)
        self._ui_ready = False
        self._log_buffer = []
        self._setup_ui_callback()
        
    def set_ui_ready(self, is_ready: bool = True):
        """
        Mark the UI as ready to receive logs and flush any buffered logs.
        
        Args:
            is_ready: Whether the UI is ready to receive logs
        """
        # Only process if the state is actually changing
        if self._ui_ready == is_ready:
            return
            
        self._ui_ready = is_ready
        
        # Only proceed if we're marking as ready and have logs to flush
        if is_ready:
            # Double-check that we have the required UI components
            if not self.ui_components or 'log_output' not in self.ui_components:
                print("[WARNING] UI marked as ready but log_output not found in components")
                return
                
            log_output = self._get_log_output_widget()
            if not log_output or not hasattr(log_output, 'append_log'):
                print("[WARNING] UI marked as ready but log output widget is not available")
                return
                
            # Now we can safely flush the buffer
            self._flush_log_buffer()
            
    def _flush_log_buffer(self):
        """
        Flush any buffered logs to the UI.
        
        This method is safe to call multiple times and will handle errors gracefully.
        It will not process logs if the UI is not ready or if there are no logs to process.
        """
        # Safety checks
        if not self._log_buffer or not self._ui_ready:
            return
            
        # Prevent re-entrancy
        if hasattr(self, '_flushing') and self._flushing:
            return
            
        self._flushing = True
        
        try:
            # Get a copy of the current buffer and clear it
            while self._log_buffer:
                # Process in chunks to avoid blocking the UI for too long
                current_chunk = self._log_buffer[:50]  # Process up to 50 logs at a time
                del self._log_buffer[:len(current_chunk)]
                
                # Process each log entry in the current chunk
                for level, message, exc_info in current_chunk:
                    try:
                        self._process_single_log(level, message, exc_info)
                    except Exception as e:
                        print(f"[ERROR] Failed to process log entry: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        
                        # If we can't process logs, put them back in the buffer
                        self._log_buffer.extend(current_chunk)
                        raise
                        
                # Small delay to prevent UI freezing
                if len(self._log_buffer) > 0:
                    import time
                    time.sleep(0.01)  # 10ms delay between chunks
                    
        except Exception as e:
            print(f"[ERROR] Failed to flush log buffer: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Always clear the flushing flag, even if an error occurred
            self._flushing = False
    
    def _process_single_log(self, level: str, message: str, exc_info=None) -> None:
        """
        Process a single log entry to be displayed in the UI.
        
        Args:
            level: Log level (debug, info, success, warning, error, critical)
            message: The log message
            exc_info: Optional exception information
        """
        try:
            # Validate UI components and log output widget
            if not self._validate_ui_components():
                return
                
            # Get the log level with fallback
            log_level = self._get_log_level(level)
            
            # Get the log output widget
            log_output = self._get_log_output_widget()
            if not log_output:
                return
            
            # Send the log to the UI
            self._send_to_ui(log_output, message, log_level)
            
            # Handle exception details if present
            if exc_info and isinstance(exc_info, BaseException):
                self._log_exception(log_output, exc_info, log_level)
                
        except Exception as e:
            self._log_processing_error(level, message, e)
    
    def _validate_ui_components(self) -> bool:
        """Validate that required UI components are available."""
        if not hasattr(self, 'ui_components') or not self.ui_components:
            print("[WARNING] UI components not initialized")
            return False
            
        if 'log_output' not in self.ui_components:
            print("[WARNING] log_output not found in UI components")
            return False
            
        return True
    
    def _get_log_level(self, level: str) -> str:
        """Get the log level with proper fallback handling."""
        try:
            from smartcash.ui.components.log_accordion import LogLevel
            level_mapping = {
                'debug': LogLevel.DEBUG,
                'info': LogLevel.INFO,
                'success': LogLevel.SUCCESS,
                'warning': LogLevel.WARNING,
                'error': LogLevel.ERROR,
                'critical': LogLevel.CRITICAL
            }
            return level_mapping.get(level.lower(), LogLevel.INFO)
            
        except (ImportError, AttributeError):
            # Fallback to string values if LogLevel import fails
            level_mapping = {
                'debug': 'debug',
                'info': 'info',
                'success': 'success',
                'warning': 'warning',
                'error': 'error',
                'critical': 'critical'
            }
            return level_mapping.get(level.lower(), 'info')
    
    def _send_to_ui(self, log_output, message: str, log_level) -> None:
        """Send a log message to the UI output widget."""
        try:
            log_output.append_log(
                message=message,
                level=log_level,
                timestamp=None  # Timestamp will be generated automatically
            )
        except Exception as e:
            print(f"[ERROR] Failed to send log to UI: {str(e)}")
            raise
    
    def _log_exception(self, log_output, exc_info, log_level) -> None:
        """Log exception details to the UI."""
        try:
            import traceback
            error_trace = ''.join(traceback.format_exception(
                type(exc_info), exc_info, exc_info.__traceback__
            ))
            self._send_to_ui(
                log_output,
                f"Exception details:\n{error_trace}",
                log_level
            )
        except Exception as e:
            print(f"[ERROR] Failed to log exception: {str(e)}")
    
    def _log_processing_error(self, level: str, message: str, error: Exception) -> None:
        """Log errors that occur during log processing."""
        error_msg = (
            f"[ERROR] Failed to process log entry. "
            f"Level: {level}, Message: {message[:100]}... Error: {str(error)}"
        )
        print(error_msg)
        import traceback
        traceback.print_exc()
    
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
    
    def _log_to_ui(self, level: str, message: str, exc_info=None) -> None:
        """Log pesan ke UI dengan error handling dan buffering."""
        # Always buffer the log first
        self._log_buffer.append((level, message, exc_info))
        
        # If UI is not ready, don't process logs yet
        if not self._ui_ready:
            return
            
        try:
            # Cek apakah komponen UI sudah siap
            if not self.ui_components or 'log_output' not in self.ui_components:
                return
                
            # Dapatkan widget output log
            log_output = self._get_log_output_widget()
            if not log_output or not hasattr(log_output, 'append_log'):
                return
                
            # If we get here, UI is ready, so flush the buffer
            self._flush_log_buffer()
            
            # Define level mapping with fallback to string values
            try:
                from smartcash.ui.components.log_accordion import LogLevel
                level_mapping = {
                    'debug': LogLevel.DEBUG,
                    'info': LogLevel.INFO,
                    'success': LogLevel.SUCCESS,
                    'warning': LogLevel.WARNING,
                    'error': LogLevel.ERROR,
                    'critical': LogLevel.CRITICAL
                }
                log_level = level_mapping.get(level.lower(), LogLevel.INFO)
            except (ImportError, AttributeError) as e:
                # Fallback to string values if LogLevel import fails
                print(f"[WARNING] Using fallback log levels: {str(e)}")
                level_mapping = {
                    'debug': 'debug',
                    'info': 'info',
                    'success': 'success',
                    'warning': 'warning',
                    'error': 'error',
                    'critical': 'critical'
                }
                log_level = level_mapping.get(level.lower(), 'info')
            
            # Panggil append_log dengan parameter yang benar
            log_output.append_log(
                message=message,
                level=log_level,
                timestamp=None  # Timestamp akan di-generate otomatis
            )
            
            # If there's an exception, log it as a separate error
            if exc_info and isinstance(exc_info, BaseException):
                import traceback
                error_trace = ''.join(traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__))
                try:
                    log_output.append_log(
                        message=f"Exception details:\n{error_trace}",
                        level=LogLevel.ERROR,
                        timestamp=None
                    )
                except NameError:
                    # Handle case where LogLevel is not available
                    log_output.append_log(
                        message=f"Exception details (ERROR):\n{error_trace}",
                        level='error',
                        timestamp=None
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to log exception: {str(e)}")
                    traceback.print_exc()
            
        except Exception as e:
            # Log the error to console if UI update fails
            print(f"[ERROR] Failed to log to UI: {str(e)}")
            print(f"[ERROR] Original message: [{level.upper()}] {message}")
            traceback.print_exc()
    
    def _get_log_output_widget(self):
        """Get the log output widget from UI components."""
        try:
            # Debug: Print available keys for troubleshooting
            available_keys = list(self.ui_components.keys())
            print(f"[DEBUG] Available UI component keys: {available_keys}")
            
            # 1. First try to get log_components
            if 'log_components' in self.ui_components and self.ui_components['log_components'] is not None:
                log_components = self.ui_components['log_components']
                if isinstance(log_components, dict) and 'log_output' in log_components:
                    log_output = log_components['log_output']
                    if hasattr(log_output, 'append_log'):
                        print("[DEBUG] Found log_output in log_components")
                        return log_output
            
            # 2. Try to get log_output directly
            if 'log_output' in self.ui_components:
                log_output = self.ui_components['log_output']
                if hasattr(log_output, 'append_log'):
                    print("[DEBUG] Found log_output directly in ui_components")
                    return log_output
            
            # 3. Try to find log_accordion and get its output
            if 'log_accordion' in self.ui_components:
                log_accordion = self.ui_components['log_accordion']
                if hasattr(log_accordion, 'children') and len(log_accordion.children) > 0:
                    log_output = log_accordion.children[0]
                    if hasattr(log_output, 'append_log'):
                        print("[DEBUG] Found log_output in log_accordion children")
                        return log_output
            
            # 4. Try to find any widget with 'log' in the key that has append_log method
            for key, widget in self.ui_components.items():
                if 'log' in key.lower() and hasattr(widget, 'append_log'):
                    print(f"[DEBUG] Found log widget with key: {key}")
                    return widget
                
                # If it's a container, check its children
                if hasattr(widget, 'children'):
                    for child in widget.children:
                        if hasattr(child, 'append_log'):
                            print(f"[DEBUG] Found log widget in children of {key}")
                            return child
            
            # 5. If we get here, log a debug message with more details
            print("[DEBUG] No suitable log output widget found. Checking widget types...")
            for key, widget in self.ui_components.items():
                print(f"[DEBUG] Widget {key}: {type(widget).__name__}, methods: {[m for m in dir(widget) if not m.startswith('_')]}")
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Error getting log output widget: {str(e)}")
            import traceback
            traceback.print_exc()
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