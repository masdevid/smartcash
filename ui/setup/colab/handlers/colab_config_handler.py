"""
file_path: smartcash/ui/setup/colab/handlers/colab_config_handler.py

Colab Configuration Handler.

This handler manages the environment setup process in Google Colab,
including configuration management and setup orchestration.
"""

from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

# Import core handlers
from smartcash.ui.core.handlers.ui_handler import ModuleUIHandler
from smartcash.ui.core.handlers import ConfigurableHandler

# Import local modules
from smartcash.ui.setup.colab.handlers.setup_handler import SetupHandler
from smartcash.ui.setup.colab.configs.config_handler import ConfigHandler
from smartcash.ui.setup.colab.constants import SetupStage

class ColabConfigHandler(ModuleUIHandler, ConfigurableHandler):
    """Main orchestrator for environment configuration in Colab.
    
    This handler coordinates various environment configuration components:
    - SetupHandler: Manages the setup workflow
    - ConfigHandler: Manages configuration (in-memory only)
    - UI updates and status management
    
    Provides a unified interface for the UI to interact with the
    environment configuration system.
    """
    
    def __init__(self, module_name: str = 'colab'):
        """Initialize the ColabConfigHandler.
        
        Args:
            module_name: Name of the module this handler manages (default: 'colab')
        """
        # Initialize base classes with required parameters
        super().__init__(module_name=module_name)  # This will call ModuleUIHandler.__init__
        ConfigurableHandler.__init__(self, module_name=module_name)
        
        # Initialize logger
        self.logger = getattr(self, 'logger', None)
        if self.logger is None:
            from smartcash.ui.logger import get_module_logger
            self.logger = get_module_logger(self.__class__.__module__)
        
        # Initialize instance variables
        self._is_initialized = False
        self.setup_handler = SetupHandler()
        self.config_handler = ConfigHandler()
        self.current_stage = SetupStage.INIT  # Updated from INITIAL to INIT to match enum
        self._callbacks = {
            'on_status_update': [],
            'on_stage_change': []
        }
        self._is_setup_complete = False
        self._setup_result = None
        
        # Set up event handlers
        self._setup_event_handlers()
        
    def _setup_event_handlers(self):
        """Set up event handlers for setup process events."""
        # OperationHandler uses a different pattern for event handling
        # We'll use the execute_operation method which handles progress and errors
        # The callbacks will be called automatically by the operation execution
        pass

    def initialize(self) -> Dict[str, Any]:
        """Initialize the handler and its components."""
        if not self._is_initialized:
            self._is_initialized = True
            self.logger.debug("✅ ColabConfigHandler initialized")
        return {"status": True, "message": "ColabConfigHandler initialized"}

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for specific events.
        
        Args:
            event_type: Type of event to register for (e.g., 'on_status_update')
            callback: Callback function to register
        """
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)

    def _notify_callbacks(self, event_type: str, *args, **kwargs) -> None:
        """Notify all registered callbacks of an event.
        
        Args:
            event_type: Type of event to notify
            *args: Positional arguments to pass to callbacks
            **kwargs: Keyword arguments to pass to callbacks
        """
        for callback in self._callbacks.get(event_type, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in {event_type} callback: {e}")

    def update_status(self, message: str, status_type: str = "info") -> None:
        """Update the status and notify listeners.
        
        Args:
            message: Status message
            status_type: Type of status (e.g., 'info', 'warning', 'error', 'success')
        """
        self._notify_callbacks('on_status_update', message, status_type)

    def set_stage(self, stage: SetupStage) -> None:
        """Set the current setup stage and notify listeners.
        
        Args:
            stage: The new setup stage
        """
        self.current_stage = stage
        self._notify_callbacks('on_stage_change', stage)

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration.
        
        Returns:
            Dictionary containing the current configuration
        """
        return self.config_handler.get_config()

    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update the configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            Updated configuration
        """
        return self.config_handler.update(updates)

    def start_setup(self) -> Dict[str, Any]:
        """Start the environment setup process.
        
        Returns:
            Dictionary with status and message
        """
        try:
            self.set_stage(SetupStage.STARTED)
            self.update_status("Starting environment setup...", "info")
            
            # Delegate to setup handler
            result = self.setup_handler.start_setup()
            
            if result.get("status"):
                self.set_stage(SetupStage.COMPLETED)
                self.update_status("Environment setup completed successfully", "success")
            else:
                self.set_stage(SetupStage.FAILED)
                self.update_status(
                    f"Environment setup failed: {result.get('message', 'Unknown error')}",
                    "error"
                )
            
            return result
            
        except Exception as e:
            self.set_stage(SetupStage.FAILED)
            error_msg = f"Error during environment setup: {str(e)}"
            self.update_status(error_msg, "error")
            self.logger.error(error_msg, exc_info=True)
            return {"status": False, "message": error_msg}

    def validate_environment(self) -> Dict[str, Any]:
        """Validate the current environment configuration.
        
        Returns:
            Dictionary with validation results
        """
        try:
            self.update_status("Validating environment...", "info")
            result = self.setup_handler.validate_environment()
            
            if result.get("status"):
                self.update_status("Environment validation passed", "success")
            else:
                self.update_status(
                    f"Environment validation failed: {result.get('message', 'Unknown error')}",
                    "warning"
                )
            
            return result
            
        except Exception as e:
            error_msg = f"Error during environment validation: {str(e)}"
            self.update_status(error_msg, "error")
            self.logger.error(error_msg, exc_info=True)
            return {"status": False, "message": error_msg}

    def reset_config(self) -> Dict[str, Any]:
        """Reset the configuration to default values.
        
        Returns:
            Dictionary with status and message
        """
        try:
            self.update_status("Resetting configuration to defaults...", "info")
            result = self.config_handler.reset_to_defaults()
            self.update_status("Configuration reset to defaults", "success")
            return result
        except Exception as e:
            error_msg = f"Error resetting configuration: {str(e)}"
            self.update_status(error_msg, "error")
            self.logger.error(error_msg, exc_info=True)
            return {"status": False, "message": error_msg}
            
    def handle_setup_button_click(self, button) -> None:
        """Handle setup button click with phase management.
        
        Args:
            button: The button widget that was clicked
        """
        try:
            # Update to in_progress phase
            if hasattr(button, 'set_phase'):
                button.set_phase('in_progress')
                
            # Disable button during processing
            if hasattr(button, 'disabled'):
                button.disabled = True
                
            # Start the setup process
            self.set_stage(SetupStage.STARTED)
            self.update_status("Memulai proses setup environment...", "info")
            
            try:
                # Execute the setup process
                result = self.start_setup()
                
                if result.get("status"):
                    # Update to completed phase on success
                    if hasattr(button, 'set_phase'):
                        button.set_phase('completed')
                    self.set_stage(SetupStage.COMPLETED)
                    self.update_status("Setup environment berhasil!", "success")
                else:
                    # Update to failed phase on error
                    if hasattr(button, 'set_phase'):
                        button.set_phase('failed')
                    self.set_stage(SetupStage.FAILED)
                    error_msg = result.get("message", "Terjadi kesalahan yang tidak diketahui")
                    self.update_status(f"Gagal setup environment: {error_msg}", "error")
                    
            except Exception as e:
                # Update to failed phase on exception
                if hasattr(button, 'set_phase'):
                    button.set_phase('failed')
                self.set_stage(SetupStage.FAILED)
                error_msg = f"Error selama setup: {str(e)}"
                self.update_status(error_msg, "error")
                self.logger.error(error_msg, exc_info=True)
                
        except Exception as e:
            error_msg = f"Error handling button click: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.update_status(error_msg, "error")
            
        finally:
            # Re-enable the button if it's in a terminal state
            if hasattr(button, 'disabled'):
                button.disabled = False

# Singleton instance for easy access
colab_config_handler = ColabConfigHandler()
