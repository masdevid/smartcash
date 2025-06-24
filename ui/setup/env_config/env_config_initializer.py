# File: smartcash/ui/setup/env_config/env_config_initializer.py 
# Deskripsi: Initialize environment configuration UI with minimal dependencies

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
# Import logger utilities
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.utils.simple_logger import create_simple_logger
from smartcash.ui.utils.logging_utils import suppress_all_outputs, restore_stdout

# Type aliases
LoggerType = Callable[[str], None]

def initialize_env_config_ui(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """Initialize environment configuration UI dengan setup sederhana"""
    # Suppress all outputs during initialization
    suppress_all_outputs()
    try:
        # 1. Create UI components first (without logging)
        ui_components = _create_ui_components()
        
        # 2. Setup logger bridge first (before any logging)
        logger_bridge = _initialize_logger_bridge(ui_components)
        ui_components['_logger_bridge'] = logger_bridge
        
        # 3. Initialize status checker if not present
        if '_status_checker' not in ui_components:
            from smartcash.ui.setup.env_config.handlers.status_checker import StatusChecker
            ui_components['_status_checker'] = StatusChecker(logger=logger_bridge)
        
        # 4. Now setup handlers (this will do the initial status update)
        _setup_handlers(ui_components, config or {})
        
        # 5. Restore output after UI is ready
        restore_stdout()
        
        # Return the container widget which holds the UI
        if 'container' in ui_components:
            return ui_components['container']
        elif 'ui' in ui_components:
            return ui_components['ui']
        else:
            raise ValueError("No valid UI container found in components")
            
    except Exception as e:
        restore_stdout()  # Ensure output is restored even on error
        error_fallback = _create_error_fallback(str(e))
        if 'container' in error_fallback:
            return error_fallback['container']
        return error_fallback
    
def _create_ui_components() -> Dict[str, Any]:
    """Create and return UI components using shared components."""
    from smartcash.ui.setup.env_config.components.ui_components import create_env_config_ui
    return create_env_config_ui()

def _update_status(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """Update status bar and log the message with appropriate status type."""
    try:
        # Validate inputs
        if not isinstance(ui_components, dict):
            raise ValueError("ui_components must be a dictionary")
        if not isinstance(message, str):
            message = str(message)
        # Normalize status type
        status_type = status_type.lower()
        # Get status bar widget
        status_bar = ui_components.get('status_bar')
        if not status_bar:
            return
        # Log the status update if logger bridge is available
        logger_bridge = ui_components.get('_logger_bridge')
        if logger_bridge and hasattr(logger_bridge, status_type):
            try:
                log_method = getattr(logger_bridge, status_type)
                log_method(f"Status: {message}")
            except Exception as log_err:
                print(f"[WARNING] Failed to log status: {str(log_err)}")
        # Map status type to colors and icons
        # Use the existing update_status_panel utility
        from smartcash.ui.setup.env_config.utils.ui_updater import update_status_panel
        update_status_panel(status_bar, message, status_type)
    except Exception as e:
        # Fallback to console if UI logging fails
        error_msg = f"[ERROR] Status update failed: {str(e)}"
        print(error_msg)
        print(f"[STATUS] {status_type.upper()}: {message}")
        # Try to log the error if possible
        if '_logger_bridge' in ui_components and hasattr(ui_components['_logger_bridge'], 'error'):
            try:
                ui_components['_logger_bridge'].error(error_msg)
            except Exception as log_err:
                print(f"[ERROR] Failed to log error: {str(log_err)}")

def _setup_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Set up handlers with proper initialization order."""
    try:
        # 1. Get the logger bridge that was already created
        logger_bridge = ui_components.get('_logger_bridge')
        if not logger_bridge:
            raise RuntimeError("Logger bridge not initialized")
            
        # 2. Initialize the setup handler
        if '_setup_handler' not in ui_components:
            from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler
            ui_components['_setup_handler'] = SetupHandler(logger=logger_bridge)
            logger_bridge.info("✅ Setup handler initialized")
            
        # 3. Get the status checker
        status_checker = ui_components.get('_status_checker')
        if not status_checker:
            logger_bridge.warning("Status checker not found, initializing...")
            from smartcash.ui.setup.env_config.handlers.status_checker import StatusChecker
            status_checker = StatusChecker(logger=logger_bridge)
            ui_components['_status_checker'] = status_checker
            
        # 3. Setup initial status
        _update_status(ui_components, "Environment configuration ready", "info")
        
        # 4. Initialize other handlers
        _setup_event_handlers(ui_components, config)
        
        # 5. Perform initial status check if status checker is available
        if status_checker:
            _perform_initial_status_check(ui_components)
        else:
            _update_status(ui_components, "Status checker not available", "warning")
        
    except Exception as e:
        error_msg = f"Error setting up handlers: {str(e)}"
        _update_status(ui_components, error_msg, "error")
        raise RuntimeError(error_msg) from e

def _setup_event_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Set up event handlers for UI components."""
    try:
        # Get the setup button from UI components
        setup_button = ui_components.get('setup_button')
        if not setup_button:
            if '_logger_bridge' in ui_components:
                ui_components['_logger_bridge'].warning("Setup button not found in UI components")
            return
            
        # Get the setup handler instance that was created earlier
        setup_handler = ui_components.get('_setup_handler')
        if not setup_handler:
            if '_logger_bridge' in ui_components:
                ui_components['_logger_bridge'].error("Setup handler not found in UI components")
            return
            
        # Setup button click handler using the setup_handler's method
        setup_button.on_click(
            lambda b: setup_handler.setup_button_handler(b, ui_components)
        )
        
        # Log successful handler setup
        if (logger := ui_components.get('_logger_bridge')):
            logger.info("✅ Setup button handler configured")
            logger.info("✅ Event handlers initialized successfully")
            
    except Exception as e:
        error_msg = f"Error setting up event handlers: {str(e)}"
        if (logger := ui_components.get('_logger_bridge')):
            logger.error(error_msg, exc_info=True)
        _update_status(ui_components, error_msg, "error")
        raise

def _create_fallback_logger(message: str = None, error: Exception = None) -> Any:
    """Create and configure a fallback logger with optional message and error handling."""
    from smartcash.ui.utils.simple_logger import create_simple_logger
    
    logger = create_simple_logger('EnvConfigLogger')
    if message:
        if error:
            logger.error(f"❌ {message}: {str(error)}", exc_info=True)
        else:
            logger.warning(f"⚠️ {message}")
    return logger

def _initialize_logger_bridge(ui_components: Dict[str, Any]) -> Any:
    """Initialize and configure the logger bridge."""
    try:
        # Initialize standard logger bridge
        logger_bridge = create_ui_logger_bridge(ui_components)
        ui_components['_logger_bridge'] = logger_bridge
        
        # Setup log output widget if it exists
        _setup_log_output_widget(ui_components)
        
        # Log initialization
        logger_bridge.info("🔌 Logger bridge initialized")
        
        return logger_bridge
        
    except Exception as e:
        # Fallback to simple logger if initialization fails
        error_msg = f"Failed to initialize logger bridge: {str(e)}"
        print(f"[ERROR] {error_msg}")
        
        # Try to create a fallback logger
        fallback_logger = _create_fallback_logger(error_msg, e)
        ui_components['_logger_bridge'] = fallback_logger
        
        return fallback_logger

def _setup_log_output_widget(ui_components: Dict[str, Any]) -> None:
    """Configure the log output widget in the UI components."""
    if 'log_components' in ui_components and isinstance(ui_components['log_components'], dict):
        log_components = ui_components['log_components']
        if 'log_output' in log_components:
            ui_components['log_output'] = log_components['log_output']

def _perform_initial_setup(ui_components: Dict[str, Any], logger_bridge) -> None:
    """Initialize handlers and perform initial status check with the provided logger bridge."""
    from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler
    from smartcash.ui.setup.env_config.handlers.status_checker import StatusChecker
    
    logger_bridge.info("🔧 Initializing environment configuration handlers...")
    
    # Initialize handlers
    setup_handler = SetupHandler(logger_bridge)
    status_checker = StatusChecker(logger_bridge)
    
    # Store components for later use
    ui_components.update({
        '_setup_handler': setup_handler,
        '_status_checker': status_checker,
        '_logger_bridge': logger_bridge
    })
    
    logger_bridge.info("✅ Handlers initialized successfully")
    
    # Perform initial status check with error handling
    try:
        _perform_initial_status_check(ui_components)
        logger_bridge.info("✅ Environment configuration handlers setup completed")
    except Exception as e:
        error_msg = f"❌ Error during initial status check: {str(e)}"
        logger_bridge.error(error_msg, exc_info=True)
        _update_status(ui_components, error_msg, "error")
        
        # Try to perform status check even if there was an error
        if '_status_checker' in ui_components:
            try:
                logger_bridge.info("🔍 Attempting status check after error...")
                _perform_initial_status_check(ui_components)
            except Exception as status_error:
                logger_bridge.error(f"❌ Failed to perform status check: {str(status_error)}", exc_info=True)

def _perform_initial_status_check(ui_components: Dict[str, Any]) -> None:
    """Check environment status and update UI components."""
    if not isinstance(ui_components, dict):
        raise ValueError("ui_components must be a dictionary")
    logger = ui_components.get('_logger_bridge', create_simple_logger('EnvConfigLogger'))
    
    try:
        logger.info("🔍 Starting initial environment status check...")
        # Update status to show we're checking
        _update_status(ui_components, "Checking environment configuration...", "info")
        # Get the status checker instance
        status_checker = ui_components.get('_status_checker')
        if not status_checker:
            raise ValueError("Status checker not initialized")
        # Update status panel to show loading state
        _update_status(ui_components, "Checking environment configuration...", "info")
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
        _update_status(ui_components, f"Error: {error_msg}", "danger")
        # Re-raise to allow caller to handle the error
        raise RuntimeError(error_msg) from e

def _create_error_fallback(error_message: str, traceback: Optional[str] = None) -> widgets.VBox:
    """Create a fallback UI component to display error messages."""
    from smartcash.ui.components import create_error_component
    return create_error_component("Initialization Error", error_message, traceback)
