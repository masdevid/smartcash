"""
File: smartcash/ui/setup/colab/colab_uimodule.py
Description: Colab Module implementation using new UIModule pattern.
"""

"""
file_path: smartcash/ui/setup/colab/colab_uimodule.py
Description: Main module for Colab UI setup and management.

This module provides the main ColabUIModule class that implements the UI for
Google Colab environment setup and management.
"""

# Standard library imports
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Callable

# Third-party imports
from IPython.display import display

# Application imports
from smartcash.ui.core.ui_module import UIModule, SharedMethodRegistry, register_operation_method
from smartcash.ui.core.ui_module_factory import create_template
from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.handlers.operation_handler import OperationHandler, OperationStatus
from smartcash.ui.core.errors.handlers import handle_ui_errors

# Colab module imports
from smartcash.ui.setup.colab.components.colab_ui import create_colab_ui
from smartcash.ui.setup.colab.configs.colab_config_handler import ColabConfigHandler
from smartcash.ui.setup.colab.configs.colab_defaults import get_default_colab_config
from smartcash.ui.setup.colab.operations.operation_manager import ColabOperationManager

# Global module instance for singleton pattern
_colab_uimodule: Optional[UIModule] = None

def register_colab_template() -> None:
    """Register Colab module template with UIModuleFactory."""
    from smartcash.ui.core.ui_module_factory import UIModuleFactory
    
    template = create_template(
        module_name="colab",
        parent_module="setup",
        default_config=get_default_colab_config(),
        required_components=["main_container", "form_container", "action_container", "operation_container"],
        required_operations=["full_setup", "init", "drive_mount", "verify"],
        auto_initialize=False,
        description="Google Colab environment setup module"
    )
    
    try:
        UIModuleFactory.register_template(template, overwrite=True)
        # Use get_module_logger locally to avoid callable errors
        local_logger = get_module_logger("smartcash.ui.setup.colab.template")
        local_logger.debug("📋 Registered Colab template")
    except Exception as e:
        local_logger = get_module_logger("smartcash.ui.setup.colab.template")
        local_logger.error(f"❌ Failed to register template: {e}")

def register_colab_shared_methods() -> None:
    """Register Colab-specific shared methods."""
    
    def mount_google_drive(drive_path: str = "/content/drive") -> Dict[str, Any]:
        """Mount Google Drive in Colab environment."""
        try:
            from google.colab import drive
            drive.mount(drive_path)
            return {"success": True, "path": drive_path}
        except ImportError:
            return {"success": False, "error": "Not running in Google Colab"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def detect_colab_environment() -> Dict[str, Any]:
        """Detect if running in Google Colab environment."""
        try:
            import google.colab
            return {"is_colab": True, "runtime_type": "colab"}
        except ImportError:
            return {"is_colab": False, "runtime_type": "local"}
    
    # Register methods with error handling for re-registration
    methods = [
        ("mount_google_drive", mount_google_drive, "Mount Google Drive"),
        ("detect_colab_environment", detect_colab_environment, "Detect Colab environment")
    ]
    
    for name, method, desc in methods:
        try:
            register_operation_method(name, method, description=desc)
        except ValueError:
            SharedMethodRegistry.register_method(name, method, overwrite=True, 
                                               description=desc, category="operations")
    
    # Use get_module_logger locally to avoid callable errors
    local_logger = get_module_logger("smartcash.ui.setup.colab.methods")
    local_logger.debug("🔗 Registered Colab shared methods")

class ColabUIModule(UIModule):
    """Colab-specific UIModule with enhanced functionality."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Colab UIModule.
        
        Args:
            config: Colab configuration (optional, uses defaults if not provided)
        """
        # Get default config and merge with provided config
        default_config = get_default_colab_config()
        if config:
            default_config.update(config)
        
        # Initialize with module name and parent module
        super().__init__(
            module_name="colab",
            parent_module="setup", 
            config=default_config,
            auto_initialize=False
        )
        
        # Initialize instance logger
        self.logger = get_module_logger(f"smartcash.ui.setup.colab.{self.__class__.__name__}")
        
        # Colab-specific attributes
        self._operation_manager: Optional[ColabOperationManager] = None
        self._config_handler: Optional[ColabConfigHandler] = None
        self._environment_detected = False
        
    def initialize(self, config: Dict[str, Any] = None) -> 'ColabUIModule':
        """Initialize Colab module with environment detection.
        
        Args:
            config: Additional configuration to merge
            
        Returns:
            Self for method chaining
        """
        if config:
            self.update_config(**config)
        
        try:
            # Detect environment first
            self._detect_environment()
            
            # Create UI components
            self._create_ui_components()
            
            # Setup operation manager
            self._setup_operation_manager()
            
            # Setup config handler (no persistence for Colab)
            self._setup_config_handler()
            
            # Register operations
            self._register_operations()
            
            # Inject shared methods
            SharedMethodRegistry.inject_methods(self, category="operations")
            
            # Setup event handlers for buttons
            self._setup_event_handlers()
            
            # Call parent initialization
            super().initialize()
            
            # Update status to show module is ready
            self._update_status("Colab module initialized - ready for environment setup", "success")
            
            self.logger.debug(f"✅ Initialized Colab UIModule")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Colab UIModule: {e}")
            self._update_status(f"Failed to initialize module: {str(e)}", "error")
            raise
        
        return self
    
    def _detect_environment(self) -> None:
        """Detect if running in Google Colab environment."""
        try:
            import google.colab
            self._environment_detected = True
            self.update_config(environment_type="colab", is_colab=True)
        except ImportError:
            self._environment_detected = False
            self.update_config(environment_type="local", is_colab=False)
    
    def _create_ui_components(self) -> None:
        """Create and register UI components."""
        try:
            # Create UI components using existing function
            ui_components = create_colab_ui(self.get_config())
            
            # Register each component
            for component_type, component in ui_components.items():
                self.register_component(component_type, component)
            
            self.logger.debug(f"Created {len(ui_components)} UI components")
            
        except Exception as e:
            self.logger.exception("Failed to create UI components")
            raise
    
    def _setup_operation_manager(self) -> None:
        """Setup operation manager for Colab operations."""
        try:
            # Get operation container instance directly
            operation_container = self.get_component("operation_container")
            
            self._operation_manager = ColabOperationManager(
                config=self.get_config(),
                operation_container=operation_container
            )
            
            self.logger.debug("Operation manager initialized")
            
        except Exception as e:
            self.logger.exception("Failed to setup operation manager")
            raise
    
    def _setup_config_handler(self) -> None:
        """Setup config handler (no persistence for Colab)."""
        try:
            self._config_handler = ColabConfigHandler()
            self.logger.debug("Config handler initialized (no persistence)")
        except Exception as e:
            self.logger.exception("Failed to setup config handler")
            raise
    
    def _register_operations(self) -> None:
        """Register Colab operations."""
        try:
            if not self._operation_manager:
                raise ValueError("Operation manager not initialized")
            
            # Get operations from manager
            operations = self._operation_manager.get_operations()
            
            # Register each operation
            for op_name, op_func in operations.items():
                self.register_operation(op_name, op_func)
            
            # Register convenience methods
            self.register_operation("status", self.get_environment_status)
            self.register_operation("reset", self.reset_environment)
            
            # Register full_setup operation if not already registered
            if 'full_setup' not in operations:
                self.register_operation('full_setup', self.execute_full_setup)
            
            self.logger.debug(f"⚙️ Registered {len(operations)} operations including full_setup")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register operations: {e}")
            raise
    
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for UI components."""
        try:
            # Get setup button and connect to execute_full_setup
            setup_button = self.get_component("setup_button") or self.get_component("primary_button")
            if setup_button:
                setup_button.on_click(self._handle_setup_button_click)
                self.logger.debug("✅ Connected setup button to handler")
            else:
                self.logger.warning("⚠️ No setup button found to connect")
            
            # Get save/reset buttons if they exist
            save_button = self.get_component("save_button")
            if save_button:
                save_button.on_click(self._handle_save_config)
                self.logger.debug("✅ Connected save button to handler")
            
            reset_button = self.get_component("reset_button")
            if reset_button:
                reset_button.on_click(self._handle_reset_config)
                self.logger.debug("✅ Connected reset button to handler")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to setup event handlers: {e}")
    
    def _handle_setup_button_click(self, button=None) -> None:
        """Handle setup button click event.
        
        Args:
            button: The button that was clicked (unused, required by ipywidgets)
        """
        try:
            # Get operation handler from parent module
            operation_handler = getattr(self, '_operation_handler', None)
            if not operation_handler:
                self.logger.error("❌ Operation handler not available")
                self._update_status("Error: Operation system not initialized", "error")
                return

            # Get operation container through the handler
            operation_container = getattr(operation_handler, '_operation_container', None)
            
            # Log start of operation
            self.logger.info("🚀 Starting environment setup")
            self._update_status("Starting environment setup...", "info")
            
            if operation_container:
                operation_container.log_message("🚀 Starting environment setup...", level='info')

            # Get button states before disabling
            button_states = self._get_operation_manager().disable_all_buttons("⏳ Setting up...")

            try:
                # Get the operation manager
                operation_manager = getattr(self, '_operation_manager', None)
                if not operation_manager:
                    raise ValueError("Operation manager not available")
                
                # Get the full_setup operation function
                full_setup_op = operation_manager.get_operations().get('full_setup')
                if not full_setup_op:
                    raise ValueError("Full setup operation not found")
                
                # Execute the operation with progress tracking
                result = self.execute_operation(
                    "full_setup",
                    full_setup_op,
                    message="Performing full environment setup..."
                )
                
                # Handle operation result (result is an OperationResult object)
                if result.status == OperationStatus.COMPLETED:
                    status_msg = "✅ Environment setup completed successfully!"
                    self._update_status(status_msg, "success")
                    self.logger.info(status_msg)
                    
                    if operation_container:
                        operation_container.log_message(status_msg, level='success')
                else:
                    error_msg = result.message or "Setup failed"
                    if result.error:
                        error_msg = f"{error_msg}: {str(result.error)}"
                    status_msg = f"❌ {error_msg}"
                    self._update_status(status_msg, "error")
                    self.logger.error(status_msg)
                    
                    if operation_container:
                        operation_container.log_message(status_msg, level='error')

                return result

            except Exception as e:
                error_msg = f"❌ Setup failed: {str(e)}"
                self.logger.error(error_msg)
                self._update_status(error_msg, "error")
                
                if operation_container:
                    operation_container.log_message(error_msg, level='error')
                    
                raise

        except Exception as e:
            self.logger.error(f"❌ Setup button click failed: {e}")
            self._update_status(f"Setup error: {str(e)}", "error")
            self.logger.error(f"❌ Setup error: {str(e)}")
            
            # Restore buttons in error state
            if button_states:
                self._get_operation_manager().enable_all_buttons(
                    button_states, 
                    success=False,
                    error_message="❌ Error"
                )
    
    def _get_operation_manager(self):
        """Get the operation manager instance for button management."""
        return self._operation_manager if self._operation_manager else self
    
    def _update_status(self, message: str, status_type: str = "info") -> None:
        """Update status panel in header container.
        
        Args:
            message: Status message to display
            status_type: Type of status ('info', 'success', 'warning', 'error')
        """
        try:
            header_container = self.get_component("header_container")
            if header_container and hasattr(header_container, 'update_status'):
                header_container.update_status(message, status_type)
                self.logger.debug(f"Updated status panel: {message} ({status_type})")
            else:
                self.logger.debug(f"Status panel not available, status: {message} ({status_type})")
        except Exception as e:
            self.logger.error(f"Failed to update status panel: {e}")
            self.logger.debug(f"Status message that failed: {message} ({status_type})")
    
    def _handle_save_config(self, button=None) -> None:
        """Handle save configuration button click.
        
        Args:
            button: The button that was clicked (unused, required by ipywidgets)
        """
        button_states = {}
        try:
            # Log using instance logger to avoid callable errors
            self.logger.info("💾 Saving configuration")
            
            # Update status
            self._update_status("Saving configuration...", "info")
            
            # Log to UI using operation container if available
            operation_container = getattr(self, '_operation_container', None)
            if operation_container:
                if hasattr(operation_container, 'log_message'):
                    operation_container.log_message("💾 Saving configuration...", level='info')
                elif hasattr(operation_container, 'log'):
                    operation_container.log("💾 Saving configuration...", level='info')
            
            # Disable buttons during save
            button_states = self._get_operation_manager().disable_all_buttons("💾 Saving...")
            
            # Get config handler
            config_handler = self.get_config_handler()
            if not config_handler:
                raise ValueError("Configuration handler not available")
            
            # Save configuration
            config_handler.save_config()
            
            # Update status and log success
            status_msg = "Configuration saved successfully"
            self._update_status(status_msg, "success")
            self.logger.info("✅ " + status_msg)
            
            # Update operation container with success
            if operation_container:
                if hasattr(operation_container, 'log_message'):
                    operation_container.log_message("✅ " + status_msg, level='success')
                elif hasattr(operation_container, 'log'):
                    operation_container.log("✅ " + status_msg, level='success')
            
            # Re-enable buttons with success state
            if button_states:
                self._get_operation_manager().enable_all_buttons(
                    button_states, 
                    success=True, 
                    success_message="💾 Saved"
                )
                
        except Exception as e:
            error_msg = f"Failed to save configuration: {str(e)}"
            self.logger.error("❌ " + error_msg)
            self._update_status(error_msg, "error")
            
            # Update operation container with error
            operation_container = getattr(self, '_operation_container', None)
            if operation_container:
                if hasattr(operation_container, 'log_message'):
                    operation_container.log_message("❌ " + error_msg, level='error')
                elif hasattr(operation_container, 'log'):
                    operation_container.log("❌ " + error_msg, level='error')
            
            # Re-enable buttons with error state
            if 'button_states' in locals() and button_states:
                self._get_operation_manager().enable_all_buttons(
                    button_states, 
                    success=False, 
                    error_message="❌ Save Failed"
                )
    
    def _handle_reset_config(self, button=None) -> None:
        """Handle reset configuration button click.
        
        Args:
            button: The button that was clicked (unused, required by ipywidgets)
        """
        button_states = {}
        try:
            # Log using instance logger to avoid callable errors
            self.logger.info("🔄 Resetting configuration to defaults")
            
            # Update status
            self._update_status("Resetting configuration to defaults...", "info")
            
            # Log to UI using operation container if available
            operation_container = getattr(self, '_operation_container', None)
            if operation_container:
                if hasattr(operation_container, 'log_message'):
                    operation_container.log_message("🔄 Resetting configuration to defaults...", level='info')
                elif hasattr(operation_container, 'log'):
                    operation_container.log("🔄 Resetting configuration to defaults...", level='info')
            
            # Disable buttons during reset
            button_states = self._get_operation_manager().disable_all_buttons("🔄 Resetting...")
            
            # Get config handler
            config_handler = self.get_config_handler()
            if not config_handler:
                raise ValueError("Configuration handler not available")
            
            # Reset configuration to defaults
            config_handler.reset_to_defaults()
            
            # Update status and log success
            status_msg = "Configuration reset to defaults successfully"
            self._update_status(status_msg, "success")
            self.logger.info("✅ " + status_msg)
            
            # Update operation container with success
            if operation_container:
                if hasattr(operation_container, 'log_message'):
                    operation_container.log_message("✅ " + status_msg, level='success')
                elif hasattr(operation_container, 'log'):
                    operation_container.log("✅ " + status_msg, level='success')
            
            # Re-enable buttons with success state
            if button_states:
                self._get_operation_manager().enable_all_buttons(
                    button_states, 
                    success=True, 
                    success_message="🔄 Reset"
                )
                
        except Exception as e:
            error_msg = f"Failed to reset configuration: {str(e)}"
            self.logger.error("❌ " + error_msg)
            self._update_status(error_msg, "error")
            
            # Update operation container with error
            operation_container = getattr(self, '_operation_container', None)
            if operation_container:
                if hasattr(operation_container, 'log_message'):
                    operation_container.log_message("❌ " + error_msg, level='error')
                elif hasattr(operation_container, 'log'):
                    operation_container.log("❌ " + error_msg, level='error')
            
            # Re-enable buttons with error state
            if 'button_states' in locals() and button_states:
                self._get_operation_manager().enable_all_buttons(
                    button_states, 
                    success=False, 
                    error_message="❌ Reset Failed"
                )
    
    def execute_full_setup(self, **kwargs) -> Dict[str, Any]:
        """Execute complete Colab setup workflow.
        
        Args:
            **kwargs: Additional arguments for setup operations
            
        Returns:
            Dictionary with setup results
        """
        if not self._operation_manager:
            return {
                "success": False,
                "error": "Operation manager not initialized",
                "message": "Cannot execute setup without operation manager"
            }
        
        try:
            # Get operation container for progress updates
            operation_container = self.get_component("operation_container")
            
            # Initialize progress
            if operation_container and hasattr(operation_container, 'update_progress'):
                operation_container.update_progress(0, "Initializing setup...", "primary")
            
            # Execute full setup operation
            result = self._operation_manager.execute_named_operation("full_setup", **kwargs)
            
            # Update final progress
            if operation_container and hasattr(operation_container, 'update_progress'):
                if result.status.value == "completed":
                    operation_container.update_progress(100, "Setup completed successfully!", "primary")
                else:
                    operation_container.update_progress(0, f"Setup failed: {result.message}", "primary")
            
            return {
                "success": result.status.value == "completed",
                "status": result.status.value,
                "message": result.message,
                "duration": result.duration,
                "data": result.data
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to execute full setup: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to execute Colab setup"
            }
    
    def is_colab_environment(self) -> bool:
        """Check if running in Google Colab environment.
        
        Returns:
            True if running in Colab, False otherwise
        """
        return self._environment_detected
    
    def get_operation_manager(self) -> Optional[ColabOperationManager]:
        """Get the operation manager instance.
        
        Returns:
            ColabOperationManager instance or None
        """
        return self._operation_manager
    
    def get_environment_status(self) -> Dict[str, Any]:
        """Get comprehensive environment status.
        
        Returns:
            Dictionary with environment information including:
            - module: Module name
            - environment_type: Type of environment (colab/local)
            - is_colab: Boolean indicating if running in Colab
            - module_status: Current module status
            - ready: Boolean indicating if module is ready
            - error_count: Number of errors encountered
            - components: Number of registered components
            - operations: Number of registered operations
            - timestamp: ISO format timestamp
        """
        return {
            "module": self.full_module_name,
            "environment_type": "colab" if self._environment_detected else "local",
            "is_colab": self._environment_detected,
            "module_status": self.get_status().value,
            "ready": self.is_ready(),
            "error_count": getattr(self, '_error_count', 0),
            "components": len(self.list_components()),
            "operations": len(self.list_operations()),
            "timestamp": datetime.now().isoformat()
        }
        
    def reset_environment(self, hard_reset: bool = False, **kwargs) -> Dict[str, Any]:
        """Reset the Colab environment to initial state.
        
        Args:
            hard_reset: If True, perform a complete reset including configs.
                        If False, perform a soft reset (reload components).
            **kwargs: Additional arguments for reset operation.
            
        Returns:
            Dictionary with reset results including:
            - success: Boolean indicating if reset was successful
            - message: Status message
            - reset_type: Type of reset performed ('hard' or 'soft')
            - timestamp: ISO format timestamp
        """
        self.logger.info(f"🔄 Resetting Colab environment (hard_reset={hard_reset})")
        
        try:
            # Get operation container for progress updates
            operation_container = getattr(self, '_operation_container', None)
            
            # Update status
            self._update_status("Resetting environment...", "info")
            
            # Log to operation container if available
            if operation_container:
                if hasattr(operation_container, 'log_message'):
                    operation_container.log_message("🔄 Resetting environment...", level='info')
                elif hasattr(operation_container, 'log'):
                    operation_container.log("🔄 Resetting environment...", level='info')
            
            # Perform hard reset if requested
            if hard_reset:
                # Reset config to defaults
                if self._config_handler:
                    self._config_handler.reset_to_defaults()
                
                # Clear any cached data
                self._clear_cached_data()
                
                # Reinitialize components
                self.initialize()
                
                message = "Environment reset to default state"
                reset_type = "hard"
            else:
                # Soft reset - just reload components
                self._create_ui_components()
                self._setup_event_handlers()
                
                message = "Environment components reloaded"
                reset_type = "soft"
            
            # Log success
            self.logger.info(f"✅ {message}")
            self._update_status(message, "success")
            
            # Update operation container with success
            if operation_container:
                if hasattr(operation_container, 'log_message'):
                    operation_container.log_message(f"✅ {message}", level='success')
                elif hasattr(operation_container, 'log'):
                    operation_container.log(f"✅ {message}", level='success')
            
            return {
                "success": True,
                "message": message,
                "reset_type": reset_type,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Failed to reset environment: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self._update_status(error_msg, "error")
            
            # Update operation container with error
            operation_container = getattr(self, '_operation_container', None)
            if operation_container:
                if hasattr(operation_container, 'log_message'):
                    operation_container.log_message(f"❌ {error_msg}", level='error')
                elif hasattr(operation_container, 'log'):
                    operation_container.log(f"❌ {error_msg}", level='error')
            
            return {
                "success": False,
                "error": str(e),
                "message": error_msg,
                "reset_type": "failed",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_config_handler(self) -> Optional[ColabConfigHandler]:
        """Get the config handler instance.
        
        Returns:
            ColabConfigHandler instance or None
        """
        return self._config_handler

def create_colab_uimodule(config: Dict[str, Any] = None, 
                         auto_initialize: bool = True,
                         force_new: bool = False) -> ColabUIModule:
    """Create Colab UIModule using factory pattern.
    
    Args:
        config: Colab configuration dictionary
        auto_initialize: Whether to auto-initialize the module
        force_new: Force creation of new instance
        
    Returns:
        ColabUIModule instance
    """
    global _colab_uimodule
    
    # Return existing instance if available and not forcing new
    if not force_new and _colab_uimodule is not None:
        if config:
            _colab_uimodule.update_config(**config)
        return _colab_uimodule
    
    try:
        # Ensure template is registered
        register_colab_template()
        
        # Ensure shared methods are registered
        register_colab_shared_methods()
        
        # Create new module instance
        module = ColabUIModule(config)
        
        # Initialize if requested
        if auto_initialize:
            module.initialize()
        
        # Store global reference
        _colab_uimodule = module
        
        # Use get_module_logger locally to avoid callable errors
        local_logger = get_module_logger("smartcash.ui.setup.colab.factory")
        local_logger.debug(f"🏭 Created Colab UIModule")
        return module
        
    except Exception as e:
        local_logger = get_module_logger("smartcash.ui.setup.colab.factory")
        local_logger.error(f"❌ Failed to create Colab UIModule: {e}")
        raise

def get_colab_uimodule(create_if_missing: bool = True, **kwargs) -> Optional[ColabUIModule]:
    """Get existing Colab UIModule instance.
    
    Args:
        create_if_missing: Create new instance if none exists
        **kwargs: Arguments for create_colab_uimodule if creating
        
    Returns:
        ColabUIModule instance or None
    """
    global _colab_uimodule
    
    if _colab_uimodule is None and create_if_missing:
        _colab_uimodule = create_colab_uimodule(**kwargs)
    
    return _colab_uimodule

def reset_colab_uimodule() -> None:
    """Reset global Colab UIModule instance."""
    global _colab_uimodule
    
    if _colab_uimodule is not None:
        try:
            _colab_uimodule.cleanup()
        except Exception as e:
            local_logger = get_module_logger("smartcash.ui.setup.colab.reset")
            local_logger.error(f"Error during cleanup: {e}")
        finally:
            _colab_uimodule = None
    
    local_logger = get_module_logger("smartcash.ui.setup.colab.reset")
    local_logger.debug("🔄 Reset global Colab UIModule instance")

# === Backward Compatibility Layer ===

@handle_ui_errors(return_type=None)
def initialize_colab_ui(config: Dict[str, Any] = None) -> None:
    """Initialize Colab UI using new UIModule pattern."""
    from IPython.display import display
    
    module = create_colab_uimodule(config, auto_initialize=True)
    main_container = module.get_component('main_container')
    if main_container:
        display(main_container)

@handle_ui_errors(return_type=dict)
def get_colab_components(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get Colab components using new UIModule pattern."""
    module = create_colab_uimodule(config, auto_initialize=True)
    return {
        component_type: module.get_component(component_type)
        for component_type in module.list_components()
    }

@handle_ui_errors(return_type=None)
def display_colab_ui(config: Dict[str, Any] = None) -> None:
    """Display Colab UI using new UIModule pattern."""
    initialize_colab_ui(config)

# Note: Template and shared methods are registered on-demand in create_colab_uimodule()
# to avoid logs during import