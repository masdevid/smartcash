"""
File: smartcash/ui/setup/colab/colab_uimodule.py
Description: Colab Module implementation using new UIModule pattern.
"""

from typing import Dict, Any, Optional
from datetime import datetime

from smartcash.ui.core.ui_module import UIModule, SharedMethodRegistry, register_operation_method
from smartcash.ui.core.ui_module_factory import create_template
from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.errors.handlers import handle_ui_errors

# Import existing Colab components and handlers
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
        
        super().__init__(
            module_name="colab",
            parent_module="setup", 
            config=default_config,
            auto_initialize=False
        )
        
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
            
            self.logger.debug(f"📦 Created {len(ui_components)} UI components")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create UI components: {e}")
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
            
            self.logger.debug("⚙️ Setup operation manager")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup operation manager: {e}")
            raise
    
    def _setup_config_handler(self) -> None:
        """Setup config handler (no persistence for Colab)."""
        try:
            self._config_handler = ColabConfigHandler()
            
            self.logger.debug("🔧 Setup config handler (no persistence)")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup config handler: {e}")
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
            
            self.logger.debug(f"⚙️ Registered {len(operations)} operations")
            
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
        """Handle setup button click."""
        button_states = {}
        try:
            # Log using instance logger to avoid callable errors
            self.logger.info("🚀 Setup button clicked - starting full setup")
            
            # Update status panel
            self._update_status("Starting environment setup...", "info")
            
            # Disable all buttons to prevent multiple clicks
            button_states = self._get_operation_manager().disable_all_buttons("⏳ Setting up environment...")
            
            # Log to UI using operation container
            self.logger.info("🚀 Starting environment setup...")
            
            # Execute full setup
            result = self.execute_full_setup()
            
            # Update status based on result
            success = result.get("success", False)
            if success:
                self._update_status("Environment setup completed successfully!", "success")
                self.log("✅ Environment setup completed successfully!", 'info')
            else:
                error_msg = result.get("message", "Setup failed")
                self._update_status(f"Setup failed: {error_msg}", "error")
                self.logger.error(f"❌ Setup failed: {error_msg}")
            
            # Restore buttons with appropriate status
            self._get_operation_manager().enable_all_buttons(
                button_states, 
                success=success,
                success_message="✅ Setup Complete" if success else "❌ Setup Failed",
                error_message="❌ Setup Failed"
            )
            
            self.logger.info(f"Setup completed with result: {result}")
            
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
                self.logger.debug(f"Status panel not available, status: {message}")
        except Exception as e:
            self.logger.error(f"Failed to update status panel: {e}")
    
    def _handle_save_config(self, button=None) -> None:
        """Handle save configuration button click."""
        button_states = {}
        try:
            # Save current configuration
            self.logger.info("💾 Saving configuration")
            self._update_status("Saving configuration...", "info")
            
            # Disable buttons during save
            button_states = self._get_operation_manager().disable_all_buttons("💾 Saving...")
            
            # Configuration is automatically saved in memory for Colab
            self._update_status("Configuration saved successfully", "success")
            self.log("✅ Configuration saved", 'info')
            
            # Re-enable buttons with success state
            self._get_operation_manager().enable_all_buttons(
                button_states, 
                success=True,
                success_message="💾 Saved"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save configuration: {e}")
            self._update_status(f"Failed to save configuration: {str(e)}", "error")
            self.log(f"❌ Failed to save configuration: {str(e)}", 'error')
            
            # Re-enable buttons with error state
            if button_states:
                self._get_operation_manager().enable_all_buttons(
                    button_states, 
                    success=False,
                    error_message="❌ Save Failed"
                )
    
    def _handle_reset_config(self, button=None) -> None:
        """Handle reset configuration button click."""
        button_states = {}
        try:
            # Reset to default configuration
            self.logger.info("🔄 Resetting configuration to defaults")
            self._update_status("Resetting configuration to defaults...", "info")
            
            # Disable buttons during reset
            button_states = self._get_operation_manager().disable_all_buttons("🔄 Resetting...")
            
            if self._config_handler:
                # Check if the method exists, use appropriate method
                if hasattr(self._config_handler, 'reset_to_defaults'):
                    self._config_handler.reset_to_defaults()
                elif hasattr(self._config_handler, 'reset_config'):
                    self._config_handler.reset_config()
                else:
                    # Fallback: manually reset using default config
                    from .configs.colab_defaults import get_default_colab_config
                    default_config = get_default_colab_config()
                    self.update_config(**default_config)
                    
            self._update_status("Configuration reset to defaults successfully", "success")
            self.log("✅ Configuration reset to defaults", 'info')
            
            # Re-enable buttons with success state
            self._get_operation_manager().enable_all_buttons(
                button_states, 
                success=True,
                success_message="🔄 Reset"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to reset configuration: {e}")
            self._update_status(f"Failed to reset configuration: {str(e)}", "error")
            self.log(f"❌ Failed to reset configuration: {str(e)}", 'error')
            
            # Re-enable buttons with error state
            if button_states:
                self._get_operation_manager().enable_all_buttons(
                    button_states, 
                    success=False,
                    error_message="❌ Reset Failed"
                )
    
    def get_environment_status(self) -> Dict[str, Any]:
        """Get comprehensive environment status.
        
        Returns:
            Dictionary with environment information
        """
        status = {
            "module": self.full_module_name,
            "environment_type": "colab" if self._environment_detected else "local",
            "is_colab": self._environment_detected,
            "module_status": self.get_status().value,
            "ready": self.is_ready(),
            "error_count": self._error_count,
            "components": len(self.list_components()),
            "operations": len(self.list_operations()),
            "timestamp": datetime.now().isoformat()
        }
        
        return status
    
    def reset_environment(self) -> Dict[str, Any]:
        """Reset Colab environment to initial state.
        
        Returns:
            Dictionary with reset results
        """
        try:
            # Reset components
            self.clear_components()
            
            # Reset operation manager
            if self._operation_manager:
                self._operation_manager.cleanup()
            
            # Re-detect environment
            self._detect_environment()
            
            # Re-create components
            self._create_ui_components()
            
            # Re-setup operation manager
            self._setup_operation_manager()
            
            # Re-register operations
            self._register_operations()
            
            return {
                "success": True,
                "message": "Colab environment reset successfully",
                "environment_type": "colab" if self._environment_detected else "local"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to reset environment: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to reset Colab environment"
            }
    
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