"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Simple initializer untuk environment configuration dengan minimal dependencies
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets

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
    # Create a simple logger first for initial setup
    temp_logger = _create_simple_logger()
    
    try:
        temp_logger.info("🔧 Initializing environment configuration handlers...")
        
        # 1. Import required modules
        from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
        from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler
        from smartcash.ui.setup.env_config.handlers.status_checker import StatusChecker
        
        # 2. Initialize the logger bridge early to capture all logs
        try:
            if 'log_output' not in ui_components:
                raise ValueError("Log output widget not found in UI components")
                
            # Create a dictionary with the log output widget
            logger_components = {
                'log_output': ui_components['log_output']
            }
            
            # Initialize the logger bridge with the correct parameters
            logger_bridge = create_ui_logger_bridge(
                ui_components=logger_components,
                logger_name='EnvConfigLogger'
            )
            temp_logger.info("✅ Logger bridge initialized successfully")
        except Exception as e:
            temp_logger.error(f"❌ Failed to initialize logger bridge: {str(e)}")
            logger_bridge = temp_logger  # Fall back to simple logger
        
        # 3. Initialize handlers with the logger bridge
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
    """📝 Create simple fallback logger"""
    class SimpleLogger:
        def info(self, msg, *args, **kwargs): print(f"ℹ️ {msg}")
        def warning(self, msg, *args, **kwargs): print(f"⚠️ {msg}")
        def error(self, msg, *args, **kwargs): 
            exc_info = kwargs.get('exc_info')
            if exc_info and exc_info != (None, None, None):
                import traceback
                print(f"❌ {msg}")
                traceback.print_exception(*exc_info)
            else:
                print(f"❌ {msg}")
        def success(self, msg, *args, **kwargs): print(f"✅ {msg}")
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
