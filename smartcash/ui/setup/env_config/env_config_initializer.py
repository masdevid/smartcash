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
        logger_bridge = create_ui_logger_bridge(ui_components)
        ui_components['_logger_bridge'] = logger_bridge
        
        # 3. Now setup handlers (this will do the initial status update)
        _setup_handlers(ui_components, config or {})
        
        # 4. Mark UI as ready to flush buffered logs
        if hasattr(logger_bridge, 'set_ui_ready'):
            logger_bridge.set_ui_ready(True)
            
        # 5. Restore output after UI is ready
        restore_stdout()
        
        return ui_components['ui']
    except Exception as e:
        restore_stdout()  # Ensure output is restored even on error
        return _create_error_fallback(str(e))
    
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
        logger_bridge = ui_components.get('logger_bridge')
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
            
        # 2. Setup initial status (will be buffered until UI is ready)
        _update_status(ui_components, "Environment configuration ready", "info")
        
        # 3. Initialize other handlers
        _setup_event_handlers(ui_components, config)
        
        # 4. Perform initial status check
        _perform_initial_status_check(ui_components)
        
    except Exception as e:
        error_msg = f"Error setting up handlers: {str(e)}"
        _update_status(ui_components, error_msg, "error")
        raise RuntimeError(error_msg) from e

def _setup_event_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Set up event handlers for UI components."""
    try:
        # Add your event handlers here
        pass
    except Exception as e:
        error_msg = f"Error setting up event handlers: {str(e)}"
        _update_status(ui_components, error_msg, "error")
        raise

def _clear_buffered_logs(ui_components: Dict[str, Any]) -> None:
    """Clear any buffered logs from the logger bridge."""
    # Import buffered logger utility
    from smartcash.ui.utils.buffered_logger import create_buffered_logger
    # Create and initialize buffered logger
    buffered_logger = create_buffered_logger()
    # Clear any existing buffered logs
    try:
        logger_bridge = ui_components.get('logger_bridge')
        if hasattr(logger_bridge, 'clear_buffer'):
            logger_bridge.clear_buffer()
    except Exception:
        # Non-critical if we can't clear the buffer
        pass
    # Store the buffered logger in ui_components immediately
    ui_components['_logger_bridge'] = buffered_logger
    try:
        # Initialize the actual UI logger bridge
        logger_bridge = None
        # Ensure we have the log_components from the UI
        if 'log_components' in ui_components and isinstance(ui_components['log_components'], dict):
            log_components = ui_components['log_components']
            # Make sure log_output is accessible directly from ui_components
            if 'log_output' in log_components:
                ui_components['log_output'] = log_components['log_output']
        # Try to find the log output widget in various possible locations
        logger_bridge = None
        if create_ui_logger_bridge is not None:
            try:
                # Pass all UI components
                logger_bridge = create_ui_logger_bridge(ui_components, 'env_config')
                ui_components['_logger_bridge'] = logger_bridge
                # Mark UI as ready to receive logs
                logger_bridge.set_ui_ready(True)
                # Flush buffered logs to the UI logger
                buffered_logger.flush_to_ui_logger(logger_bridge)
                logger_bridge.info("✅ Logger bridge initialized successfully")
            except Exception as e:
                # Fallback to simple logger if UI logger initialization fails
                logger_bridge = create_simple_logger('EnvConfigLogger')
                logger_bridge.error(f"❌ Failed to initialize UI logger bridge: {str(e)}")
                # Flush any buffered logs to the simple logger
                buffered_logger.flush_to_ui_logger(logger_bridge)
        else:
            # Fallback to simple logger if UI components are not available
            logger_bridge = create_simple_logger('EnvConfigLogger')
            logger_bridge.warning("UI logger bridge or log output widget not available, using fallback logger")
            # Flush any buffered logs to the simple logger
            buffered_logger.flush_to_ui_logger(logger_bridge)
        # Store the logger bridge in ui_components
        ui_components['_logger_bridge'] = logger_bridge
    except Exception as e:
        # Fall back to simple logger if UI logger initialization fails
        simple_logger = create_simple_logger('EnvConfigLogger')
        simple_logger.error(f"❌ Failed to initialize UI logger bridge: {str(e)}")
        # Flush any buffered logs to the simple logger
        buffered_logger.flush_to_ui_logger(simple_logger)
        # Replace the buffered logger with the simple logger
        ui_components['_logger_bridge'] = simple_logger
        logger_bridge = simple_logger
    # Now that we have a logger, use it for all logging
    logger_bridge.info("🔧 Initializing environment configuration handlers...")
    # Import required modules
    from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler
    from smartcash.ui.setup.env_config.handlers.status_checker import StatusChecker
    # Initialize handlers with the logger bridge
    setup_handler = SetupHandler(logger_bridge)
    status_checker = StatusChecker(logger_bridge)
    logger_bridge.info("✅ Handlers initialized successfully")
    # Connect setup button click handler
    def on_setup_button_clicked(button):
        logger_bridge.info("🚀 Starting environment setup...")
        try:
            setup_handler.run_full_setup(ui_components)
        except Exception as e:
            error_msg = f"❌ Error during setup: {str(e)}"
            logger_bridge.error(error_msg, exc_info=True)
            _update_status(ui_components, error_msg, "error")
    # Store components for later use
    ui_components.update({
        '_setup_handler': setup_handler,
        '_status_checker': status_checker,
        '_logger_bridge': logger_bridge
    })
    # Connect the button click handler
    if 'setup_button' in ui_components:
        ui_components['setup_button'].on_click(on_setup_button_clicked)
    else:
        logger_bridge.warning("Setup button not found in UI components")
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
                ui_components['_logger_bridge'].info("🔍 Attempting status check after error...")
                _perform_initial_status_check(ui_components)
            except Exception as status_error:
                ui_components['_logger_bridge'].error(f"❌ Failed to perform status check: {str(status_error)}", exc_info=True)

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
