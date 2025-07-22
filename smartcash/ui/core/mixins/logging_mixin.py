"""
Logging mixin for UI modules.

Provides standard logging functionality with operation container integration.
"""

from typing import Dict, Any, Optional
# Removed problematic import for now


class LoggingMixin:
    """
    Mixin providing common logging functionality.
    
    This mixin provides:
    - Operation container logging integration
    - Fallback to standard logger
    - UI logging bridge setup
    - Standard log message formatting
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ui_logging_bridge_setup: bool = False
        self._log_buffer: list = []  # Buffer logs until operation container is ready
        self._module_namespace: Optional[str] = None
        
        # Initialize logger if not already set by subclass
        if not hasattr(self, 'logger'):
            self._initialize_logger()
    
    def _initialize_logger(self) -> None:
        """Initialize the logger for this module."""
        try:
            from smartcash.ui.logger import get_module_logger
            
            # Determine module name for logger
            module_name = 'smartcash.ui.core'  # default
            if hasattr(self, 'full_module_name'):
                module_name = self.full_module_name
            elif hasattr(self, 'module_name'):
                module_name = self.module_name
            else:
                # Try to infer from class name
                class_name = self.__class__.__name__.lower()
                if 'preprocessing' in class_name:
                    module_name = 'dataset.preprocessing'
                elif 'visualization' in class_name:
                    module_name = 'dataset.visualization'
                elif 'colab' in class_name:
                    module_name = 'setup.colab'
                # Add other module patterns as needed
            
            self.logger = get_module_logger(module_name)
            
        except Exception as e:
            # Fallback to basic logger
            import logging
            self.logger = logging.getLogger(module_name)
    
    def _get_module_namespace(self) -> str:
        """
        Get the appropriate namespace for this module's logs.
        
        Returns:
            Namespace string that matches the log_namespace_filter
        """
        # Return cached namespace if available
        if hasattr(self, '_module_namespace') and self._module_namespace:
            return self._module_namespace
            
        # Initialize _module_namespace if not set
        if not hasattr(self, '_module_namespace'):
            self._module_namespace = None
            
        # Check if module has explicit namespace info
        if hasattr(self, 'module_name'):
            self._module_namespace = self.module_name
            return self._module_namespace
        
        # Check if this is a BaseUIModule with module_name
        if hasattr(self, 'full_module_name'):
            self._module_namespace = self.full_module_name
            return self._module_namespace
        
        # Try to infer from class name
        class_name = self.__class__.__name__.lower()
        if 'colab' in class_name:
            self._module_namespace = 'colab'
        elif 'downloader' in class_name:
            self._module_namespace = 'downloader'
        elif 'split' in class_name:
            self._module_namespace = 'split'
        elif 'preprocessing' in class_name:
            self._module_namespace = 'preprocessing'
        elif 'dependency' in class_name:
            self._module_namespace = 'dependency'
        elif 'augmentation' in class_name:
            self._module_namespace = 'augmentation'
        elif 'pretrained' in class_name:
            self._module_namespace = 'pretrained'
        elif 'visualization' in class_name:
            self._module_namespace = 'visualization'
        elif 'backbone' in class_name:
            self._module_namespace = 'backbone'
        elif 'training' in class_name:
            self._module_namespace = 'training'
        elif 'train' in class_name:
            self._module_namespace = 'train'
        elif 'evaluation' in class_name:
            self._module_namespace = 'evaluation'
        else:
            # Default fallback
            self._module_namespace = 'smartcash.ui.core'
            
        return self._module_namespace
        
    def _update_logging_context(self) -> None:
        """
        Update the logging context with the current module information.
        This should be called after the module is fully initialized.
        """
        # Clear any cached namespace to force recalculation
        if hasattr(self, '_module_namespace'):
            del self._module_namespace
            
        # Get the updated namespace
        namespace = self._get_module_namespace()
        
        # Update the logger if it exists
        if hasattr(self, 'logger') and hasattr(self.logger, 'set_namespace'):
            self.logger.set_namespace(namespace)
    
    def log(self, message: str, level: str = 'info') -> None:
        """
        Log message to operation container or fallback to logger.
        
        Args:
            message: Message to log
            level: Log level (info, warning, error, debug)
        """
        try:
            # Check if we should buffer logs (operation container not ready yet)
            if hasattr(self, '_log_buffer') and hasattr(self, '_is_initialized'):
                operation_container = None
                
                # Check for operation container availability
                if hasattr(self, '_ui_components') and self._ui_components:
                    operation_container = self._ui_components.get('operation_container')
                
                # If operation container isn't ready but we're initialized, buffer the log
                if not operation_container and self._is_initialized:
                    self._log_buffer.append((message, level))
                    return
            
            # Determine namespace for this module
            namespace = self._get_module_namespace()
            
            # Try operation container directly
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container:
                    # Convert string level to LogLevel enum if needed
                    ui_level = self._convert_to_ui_level(level)
                    
                    # Handle dict-style operation container
                    if isinstance(operation_container, dict):
                        if 'log_message' in operation_container:
                            operation_container['log_message'](message, ui_level, namespace)
                            return
                        elif 'log' in operation_container:  # For backward compatibility
                            operation_container['log'](message, ui_level, namespace)
                            return
                    # Handle object-style operation container
                    elif hasattr(operation_container, 'log_message'):
                        operation_container.log_message(message, ui_level, namespace)
                        return
                    elif hasattr(operation_container, 'log'):  # For backward compatibility
                        operation_container.log(message, ui_level, namespace)
                        return
            
            # Fallback to standard logger (use debug to minimize console output)
            if hasattr(self, 'logger'):
                self.logger.debug(f"[{level.upper()}] {message}")
                
        except Exception as e:
            # Final fallback (use debug to minimize console output)
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to log message: {e}")
                level_str = level.name.lower() if hasattr(level, 'name') else str(level).lower()
                self.logger.debug(f"[{level_str.upper()}] {message}")
            else:
                # Suppress print during normal operation to avoid console spam
                pass
    
    def _convert_to_ui_level(self, level: str):
        """Convert string level to LogLevel enum for UI components."""
        try:
            from smartcash.ui.components.log_accordion import LogLevel
            
            level_mapping = {
                'debug': LogLevel.DEBUG,
                'info': LogLevel.INFO,
                'warning': LogLevel.WARNING,
                'error': LogLevel.ERROR,
                'critical': LogLevel.ERROR,
                'success': LogLevel.INFO  # Map success to info level
            }
            
            return level_mapping.get(level.lower(), LogLevel.INFO)
        except ImportError:
            return level  # Return original level if LogLevel not available
    
    def log_exception(self, message: str, exception: Exception) -> None:
        """Log exception with full traceback.
        
        Args:
            message: Context message
            exception: Exception object
        """
        import traceback
        
        # Get full traceback
        tb_str = traceback.format_exc()
        
        # Format message with traceback
        error_with_traceback = f"{message}\n\nError: {str(exception)}\n\n{tb_str}"
        self.log(f"❌ {error_with_traceback}", 'error')
    
    def _setup_ui_logging_bridge(self, operation_container: Any) -> None:
        """
        Setup UI logging bridge to capture backend service logs and configure UILogger.
        
        Args:
            operation_container: Operation container to bridge logs to
        """
        try:
            if getattr(self, '_ui_logging_bridge_setup', False):
                return
                
            # CRITICAL: Setup operation container logging bridge FIRST
            # This must happen before any backend operations begin
            if hasattr(operation_container, 'setup_logging_bridge'):
                operation_container.setup_logging_bridge()
                self.log_debug("✅ Operation container logging bridge activated")
                
            # CRITICAL: Activate SmartCashLogger UI mode globally
            # This prevents backend services from logging to console
            try:
                from smartcash.common.logger import SmartCashLogger
                # Get UI handler from operation container for SmartCashLogger
                if hasattr(operation_container, '_ui_handler'):
                    SmartCashLogger.set_ui_mode(True, operation_container._ui_handler)
                    self.log_debug("✅ SmartCashLogger UI mode activated globally")
                else:
                    # Fallback: just activate UI mode to disable console
                    SmartCashLogger.set_ui_mode(True, None)
                    self.log_debug("✅ SmartCashLogger console output disabled")
            except ImportError:
                self.log_debug("SmartCashLogger not available for UI mode activation")
            except Exception as e:
                self.log_debug(f"Failed to activate SmartCashLogger UI mode: {e}")
            
            # Capture module-specific loggers
            if hasattr(self, 'logger') and hasattr(operation_container, 'capture_logs'):
                operation_container.capture_logs(self.logger)
                self.log_debug("✅ Module logger captured")
            
            # Proactively capture ALL backend service loggers before they're used
            self._capture_backend_loggers(operation_container)
            
            # Configure UILogger to use operation container (universal solution)
            if hasattr(self, 'logger') and hasattr(self.logger, 'ui_components'):
                self._configure_uilogger_bridge(operation_container)
                
            self._ui_logging_bridge_setup = True
            
            # Final verification
            self.log_debug("✅ UI logging bridge setup complete - backend logs will be captured")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to setup UI logging bridge: {e}")
    
    def _capture_backend_loggers(self, operation_container: Any) -> None:
        """
        Proactively capture ALL backend service loggers before they're used.
        
        This is critical to prevent logging leaks - we capture loggers for all
        backend services that might be called during operations.
        
        Args:
            operation_container: Operation container to capture logs to
        """
        try:
            import logging
            
            if not hasattr(operation_container, 'capture_logs'):
                return
            
            # Define backend logger patterns to capture based on module type
            backend_patterns = []
            
            # Determine module type and capture appropriate backend loggers
            if hasattr(self, 'module_name'):
                module_name = self.module_name.lower()
                
                if 'preprocessing' in module_name:
                    backend_patterns.extend([
                        'smartcash.dataset.preprocessor',
                        'smartcash.dataset.preprocessor.api',
                        'smartcash.dataset.preprocessor.service',
                        'smartcash.dataset.preprocessor.api.preprocessing_api',
                        'smartcash.dataset.preprocessor.core',
                        'smartcash.dataset.preprocessor.utils',
                        'smartcash.dataset.preprocessor.validation'
                    ])
                elif 'augmentation' in module_name:
                    backend_patterns.extend([
                        'smartcash.dataset.augmentor',
                        'smartcash.dataset.augmentor.api',
                        'smartcash.dataset.augmentor.service',
                        'smartcash.dataset.augmentor.core',
                        'smartcash.dataset.augmentor.utils'
                    ])
                elif 'backbone' in module_name:
                    backend_patterns.extend([
                        'smartcash.model.backbone',
                        'smartcash.model.backbone.api',
                        'smartcash.model.backbone.service',
                        'smartcash.model.backbone.builder'
                    ])
                elif 'training' in module_name:
                    backend_patterns.extend([
                        'smartcash.model.trainer',
                        'smartcash.model.trainer.api',
                        'smartcash.model.trainer.service',
                        'smartcash.model.trainer.core'
                    ])
                elif 'evaluation' in module_name:
                    backend_patterns.extend([
                        'smartcash.model.evaluator',
                        'smartcash.model.evaluator.api',
                        'smartcash.model.evaluator.service'
                    ])
                elif 'visualization' in module_name:
                    backend_patterns.extend([
                        'smartcash.dataset.visualizer',
                        'smartcash.dataset.visualizer.api',
                        'smartcash.dataset.visualizer.service'
                    ])
            
            # Always capture common SmartCash loggers
            backend_patterns.extend([
                'smartcash',
                'smartcash.common',
                'smartcash.common.logger',
                'smartcash.dataset',
                'smartcash.model'
            ])
            
            # Capture loggers by pattern
            captured_count = 0
            for pattern in backend_patterns:
                try:
                    backend_logger = logging.getLogger(pattern)
                    operation_container.capture_logs(backend_logger)
                    captured_count += 1
                except Exception as e:
                    self.log_debug(f"Could not capture logger '{pattern}': {e}")
            
            self.log_debug(f"✅ Captured {captured_count} backend loggers proactively")
            
            # Also capture any existing SmartCash loggers that might already be initialized
            try:
                for name, logger in logging.Logger.manager.loggerDict.items():
                    if isinstance(logger, logging.Logger) and 'smartcash' in name.lower():
                        operation_container.capture_logs(logger)
                        self.log_debug(f"✅ Captured existing logger: {name}")
            except Exception as e:
                self.log_debug(f"Could not capture existing SmartCash loggers: {e}")
                
        except Exception as e:
            self.log_debug(f"Failed to capture backend loggers: {e}")
    
    def _configure_uilogger_bridge(self, operation_container: Any) -> None:
        """
        Configure UILogger to properly bridge with operation container.
        
        Args:
            operation_container: Operation container to bridge to
        """
        try:
            import logging
            
            # Add operation container as log_output for UILogger
            if not self.logger.ui_components:
                self.logger.ui_components = {}
            
            # Create adapter for operation container log method to match UILogger interface
            if isinstance(operation_container, dict) and 'log' in operation_container:
                class LogOutputAdapter:
                    def __init__(self, log_func):
                        self.log_func = log_func
                    
                    def append_stdout(self, message):
                        # Parse emoji and level from message
                        if message.startswith('✅'):
                            level = 'success'
                        elif message.startswith('❌'):
                            level = 'error'
                        elif message.startswith('⚠️'):
                            level = 'warning'
                        elif message.startswith('🔍'):
                            level = 'debug'
                        else:
                            level = 'info'
                        
                        # Clean message (remove emoji and newlines)
                        clean_message = message.strip()
                        if clean_message.startswith(('✅', '❌', '⚠️', '🔍', 'ℹ️')):
                            clean_message = clean_message[2:].strip()
                        
                        self.log_func(clean_message, level)
                
                # Add both modern operation_container and legacy log_output for compatibility
                self.logger.ui_components['operation_container'] = operation_container
                self.logger.ui_components['log_output'] = LogOutputAdapter(operation_container['log'])  # Legacy compatibility
                
                # Suppress console logging when operation container is available
                if hasattr(self.logger, 'logger') and self.logger.logger.handlers:
                    for handler in self.logger.logger.handlers[:]:
                        if hasattr(handler, 'stream'):
                            # Disable console handler when operation container is active
                            handler.setLevel(logging.CRITICAL + 1)  # Effectively disable
                            
                if hasattr(self, 'logger'):
                    self.logger.debug("✅ UILogger bridge configured for operation container")
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to configure UILogger bridge: {e}")
    
    def log_info(self, message: str) -> None:
        """Log info message."""
        self.log(message, 'info')
    
    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.log(message, 'warning')
    
    def log_error(self, message: str, traceback: str = None) -> None:
        """Log error message with optional traceback.
        
        Args:
            message: Error message
            traceback: Optional traceback string
        """
        if traceback:
            # Format error with traceback for expansion
            error_with_traceback = f"{message}\n\n{traceback}"
            self.log(f"❌ {error_with_traceback}", 'error')
        else:
            self.log(f"❌ {message}", 'error')
    
    def log_debug(self, message: str) -> None:
        """Log debug message."""
        self.log(message, 'debug')
    
    def log_success(self, message: str) -> None:
        """Log success message."""
        self.log(f"✅ {message}", 'info')
    
    def log_operation_start(self, operation_name: str) -> None:
        """Log operation start."""
        self.log(f"🔄 Starting {operation_name}...", 'info')
    
    def log_operation_complete(self, operation_name: str) -> None:
        """Log operation completion."""
        self.log(f"✅ {operation_name} completed", 'info')
    
    def log_operation_error(self, operation_name: str, error: str, traceback: str = None) -> None:
        """Log operation error with optional traceback.
        
        Args:
            operation_name: Name of the operation that failed
            error: Error message
            traceback: Optional traceback string
        """
        if traceback:
            # Format error with traceback for expansion
            error_with_traceback = f"{error}\n\n{traceback}"
            self.log(f"❌ {operation_name} failed: {error_with_traceback}", 'error')
        else:
            self.log(f"❌ {operation_name} failed: {error}", 'error')
    
    def clear_logs(self) -> None:
        """Clear operation container logs."""
        try:
            if self._operation_manager and hasattr(self._operation_manager, 'clear_logs'):
                self._operation_manager.clear_logs()
            elif hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container and hasattr(operation_container, 'clear_logs'):
                    operation_container.clear_logs()
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to clear logs: {e}")
    
    def get_logs(self) -> Optional[str]:
        """
        Get current logs from operation container.
        
        Returns:
            Current logs or None if not available
        """
        try:
            if self._operation_manager and hasattr(self._operation_manager, 'get_logs'):
                return self._operation_manager.get_logs()
            elif hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container and hasattr(operation_container, 'get_logs'):
                    return operation_container.get_logs()
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to get logs: {e}")
        
        return None
    
    def _flush_log_buffer(self) -> None:
        """Flush buffered logs to operation container when it becomes available."""
        try:
            if hasattr(self, '_log_buffer') and self._log_buffer:
                # Get operation container
                operation_container = None
                if hasattr(self, '_ui_components') and self._ui_components:
                    operation_container = self._ui_components.get('operation_container')
                
                if operation_container:
                    # Get namespace for this module
                    namespace = self._get_module_namespace()
                    
                    # Flush all buffered logs
                    for message, level in self._log_buffer:
                        if isinstance(operation_container, dict) and 'log' in operation_container:
                            operation_container['log'](message, level, namespace)
                        elif hasattr(operation_container, 'log'):
                            operation_container.log(message, level, namespace)
                    
                    # Clear buffer after flushing
                    self._log_buffer.clear()
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to flush log buffer: {e}")