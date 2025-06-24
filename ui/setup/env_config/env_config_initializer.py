"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Simple initializer untuk environment configuration dengan minimal dependencies
"""

from typing import Dict, Any, Optional, Tuple, Type, Union, List
import ipywidgets as widgets
import traceback
import sys

# Import logger bridge with error handling
try:
    from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
except ImportError as e:
    print(f"⚠️ Could not import logger_bridge: {e}")
    create_ui_logger_bridge = None

def initialize_env_config_ui(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """
    🚀 Initialize environment configuration UI dengan setup sederhana
    
    Args:
        config: Konfigurasi opsional untuk setup
        
    Returns:
        UI container siap pakai
    """
    try:
        # 1. Create UI components
        ui_components = _create_ui_components()
        
        # 2. Setup basic handlers (includes initial status check)
        _setup_handlers(ui_components, config or {})
        
        return ui_components['ui']
        
    except Exception as e:
        return _create_error_fallback(str(e))

def _create_ui_components() -> Dict[str, Any]:
    """🎨 Buat komponen UI dengan shared components"""
    from smartcash.ui.setup.env_config.components.ui_components import create_env_config_ui
    return create_env_config_ui()

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """
    Update status panel with proper error handling and fallback
    
    Args:
        ui_components: Dictionary of UI components
        message: Message to display
        status_type: Type of status (info, success, warning, danger)
    """
    if 'status_panel' not in ui_components:
        return
        
    try:
        from smartcash.ui.setup.env_config.utils.ui_updater import update_status_panel
        update_status_panel(
            status_widget=ui_components['status_panel'],
            message=message,
            status_type=status_type
        )
    except Exception as e:
        # Fallback to direct assignment if update_status_panel fails
        try:
            ui_components['status_panel'].value = f"<div style='color: red;'>{message}</div>"
            if '_logger_bridge' in ui_components:
                ui_components['_logger_bridge'].error(f"Failed to update status panel: {str(e)}")
        except:
            pass  # If everything fails, just continue

def _setup_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Setup handlers and logger bridge with proper initialization order
    
    Args:
        ui_components: Dictionary of UI components to update
        config: Configuration dictionary for the environment setup
    """
    # Initialize logger bridge first to capture all logs
    logger_bridge = None
    
    # Check if we can use the UI logger bridge
    if create_ui_logger_bridge is not None:
        try:
            if 'log_output' not in ui_components:
                raise ValueError("Log output widget not found in UI components")
                
            # Initialize the logger bridge with the correct parameters
            logger_bridge = create_ui_logger_bridge(
                ui_components={'log_output': ui_components['log_output']},
                logger_name='EnvConfigLogger'
            )
            logger_bridge.info("✅ Logger bridge initialized successfully")
            
        except Exception as e:
            # Create simple logger for the error message
            temp_logger = _create_simple_logger()
            temp_logger.error(f"❌ Failed to initialize UI logger bridge: {str(e)}")
            logger_bridge = temp_logger
    else:
        # create_ui_logger_bridge is None, use simple logger
        logger_bridge = _create_simple_logger()
        logger_bridge.warning("⚠️ UI logger bridge not available, using console logger")
    
    # Store logger bridge in ui_components for later use
    ui_components['_logger_bridge'] = logger_bridge
    
    # Now that we have a logger, use it for all logging
    logger_bridge.info("🔧 Initializing environment configuration handlers...")
    
    # Import required modules after logger is set up
    from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler
    from smartcash.ui.setup.env_config.handlers.status_checker import StatusChecker
    
    # 2. Initialize handlers with the logger bridge
    try:
        # Import required modules
        from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler
        from smartcash.ui.setup.env_config.handlers.status_checker import StatusChecker
        try:
            setup_handler = SetupHandler(logger_bridge)
            status_checker = StatusChecker(logger_bridge)
            logger_bridge.info("✅ Handlers initialized successfully")
        except Exception as e:
            logger_bridge.error(f"❌ Failed to initialize handlers: {str(e)}", exc_info=True)
            raise
        
        # 4. Store components for later use
        ui_components.update({
            '_setup_handler': setup_handler,
            '_status_checker': status_checker,
            '_logger_bridge': logger_bridge
        })
        
        # 5. Perform initial status check with error handling
        _perform_initial_status_check(ui_components)
        
        logger_bridge.info("✅ Environment configuration handlers setup completed")
        
    except Exception as e:
        error_msg = f"❌ Critical error in handler setup: {str(e)}"
        
        # Try to log the error
        try:
            if '_logger_bridge' in ui_components:
                ui_components['_logger_bridge'].error(error_msg, exc_info=True)
            else:
                temp_logger.error(error_msg, exc_info=True)
        except:
            pass  # If logging fails, we'll continue with the error handling
        
        # Update status panel with error
        _update_status_panel(ui_components, f"Error: {error_msg}", "danger")
        
        # Try to perform status check even if there was an error
        try:
            if '_status_checker' in ui_components:
                ui_components['_logger_bridge'].info("🔍 Attempting status check after error...")
                _perform_initial_status_check(ui_components)
        except Exception as status_error:
            if '_logger_bridge' in ui_components:
                ui_components['_logger_bridge'].error(f"❌ Status check failed: {str(status_error)}")

def _perform_initial_status_check(ui_components: Dict[str, Any]) -> None:
    """
    Perform initial environment status check and update UI components
    
    Args:
        ui_components: Dictionary of UI components to update
    """
    logger = ui_components.get('_logger_bridge', _create_simple_logger())
    
    try:
        logger.info("🔍 Starting initial environment status check...")
        
        # Get the status checker instance
        status_checker = ui_components.get('_status_checker')
        if not status_checker:
            raise ValueError("Status checker not initialized")
        
        # Update status panel to show loading state
        _update_status_panel(ui_components, "Checking environment configuration...", "info")
        
        # Perform the status check
        status_result = status_checker.check_initial_status(ui_components)
        
        # Process the result
        if not isinstance(status_result, dict):
            raise ValueError("Invalid status check result format")
        
        # Update status panel with the result
        status_type = status_result.get('status_type', 'info')
        status_msg = status_result.get('status_message', 'No status message available')
        
        logger.info(f"✅ Initial status check completed: {status_msg}")
        
        # Update setup summary if available
        if 'setup_summary' in ui_components:
            try:
                from smartcash.ui.setup.env_config.components.setup_summary import update_setup_summary
                update_setup_summary(
                    ui_components['setup_summary'],
                    status_msg,
                    status_type,
                    details=status_result.get('env_info', {})
                )
                logger.debug("Updated setup summary with status check results")
            except Exception as e:
                logger.warning(f"Could not update setup summary: {str(e)}", exc_info=True)
        
        return status_result
        
    except Exception as e:
        error_msg = f"❌ Failed to perform initial status check: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update status panel with error
        _update_status_panel(ui_components, f"Error: {error_msg}", "danger")
        
        # Re-raise to allow caller to handle the error
        raise RuntimeError(error_msg) from e

def _create_simple_logger():
    """📝 Create simple fallback logger with all required methods"""
    class SimpleLogger:
        def __init__(self, name: str = 'SimpleLogger'):
            self.name = name
            self.level = 'INFO'
            
        def _log(self, level: str, msg: str, *args, **kwargs):
            """Centralized logging method"""
            prefix = {
                'debug': '🐛',
                'info': 'ℹ️',
                'warning': '⚠️',
                'error': '❌',
                'critical': '🔥',
                'success': '✅'
            }.get(level.lower(), '📝')
            
            # Handle exc_info if present
            exc_info = kwargs.pop('exc_info', None)
            if exc_info and exc_info != (None, None, None):
                if isinstance(exc_info, BaseException):
                    exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
                elif not isinstance(exc_info, tuple):
                    exc_info = sys.exc_info()
                
                # Print the message first
                print(f"{prefix} {msg}")
                # Then print the traceback
                traceback.print_exception(exc_info[0], exc_info[1], exc_info[2], file=sys.stderr)
            else:
                print(f"{prefix} {msg}")
        
        # Standard logging methods
        def debug(self, msg: str, *args, **kwargs):
            self._log('debug', msg, *args, **kwargs)
            
        def info(self, msg: str, *args, **kwargs):
            self._log('info', msg, *args, **kwargs)
            
        def warning(self, msg: str, *args, **kwargs):
            self._log('warning', msg, *args, **kwargs)
            
        def error(self, msg: str, *args, **kwargs):
            self._log('error', msg, *args, **kwargs)
            
        def critical(self, msg: str, *args, **kwargs):
            self._log('critical', msg, *args, **kwargs)
            
        def success(self, msg: str, *args, **kwargs):
            self._log('success', msg, *args, **kwargs)
            
        def exception(self, msg: str, *args, **kwargs):
            kwargs['exc_info'] = True
            self._log('error', msg, *args, **kwargs)
    
    return SimpleLogger()

def _create_error_fallback(error_message: str, traceback: Optional[str] = None) -> widgets.VBox:
    """❌ Create error fallback UI with optional traceback
    
    Args:
        error_message: The error message to display
        traceback: Optional traceback information
        
    Returns:
        A widget containing the error message and optional traceback
    """
    from smartcash.ui.components import create_error_component
    
    # Create error component with optional traceback
    error_component = create_error_component(
        error_message=error_message,
        traceback=traceback,
        title="Environment Setup Error",
        error_type="error"
    )
    
    return error_component['container']
