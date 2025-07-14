"""
File: smartcash/ui/setup/colab/colab_uimodule.py
Description: Colab Module implementation using new UIModule pattern.
"""

import os
import sys
import logging
from typing import Any, Dict, Optional, Callable, List, Tuple, Union

# Import IPython display if available
try:
    from IPython.display import display, clear_output
except ImportError:
    display = print
    clear_output = lambda: print("\n" * 100)  # Simple clear for non-IPython environments

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
from smartcash.ui.core.decorators import handle_ui_errors
from smartcash.ui.core.decorators import suppress_ui_init_logs

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
        
    @suppress_ui_init_logs(duration=3.0)
    def initialize(self, config: Dict[str, Any] = None) -> 'ColabUIModule':
        """Initialize Colab module with environment detection and setup all components.
        
        Args:
            config: Additional configuration to merge
            
        Returns:
            Self for method chaining
            
        Raises:
            RuntimeError: If any critical initialization step fails
        """
        try:
            self.logger.info("🚀 Initializing Colab UIModule...")
            
            # Update config if provided
            if config:
                self.logger.debug("Updating module configuration")
                self.update_config(**config)
            
            # 1. Detect environment first
            self.logger.debug("Detecting environment...")
            self._detect_environment()
            self.logger.info(f"Environment detected: {'Colab' if self._environment_detected else 'Local'}")
            
            # 2. Initialize parent class
            self.logger.debug("Initializing parent UIModule...")
            super().initialize()
            
            # 3. Create UI components
            self.logger.debug("Creating UI components...")
            self._create_ui_components()
            
            # 4. Setup operation manager (must be before operation registration)
            self.logger.debug("Setting up operation manager...")
            self._setup_operation_manager()
            
            # 5. Setup config handler
            self.logger.debug("Setting up config handler...")
            self._setup_config_handler()
            
            # 6. Register operations
            self.logger.debug("Registering operations...")
            self._register_operations()
            
            # 7. Inject shared methods
            self.logger.debug("Injecting shared methods...")
            SharedMethodRegistry.inject_methods(self, category="operations")
            
            # 8. Setup event handlers (must be after UI components are created)
            self.logger.debug("Setting up event handlers...")
            self._setup_event_handlers()
            
            # Verify all components are properly initialized
            self._verify_initialization()
            
            # Log initialization completion to operation container
            self._log_initialization_complete()
            
            # Update status to show module is ready
            status_msg = "✅ Colab module initialized - ready for environment setup"
            self._update_status(status_msg, "success")
            self.logger.info(status_msg)
            
            return self
            
        except Exception as e:
            error_msg = f"❌ Failed to initialize Colab UIModule: {str(e)}"
            self.logger.exception(error_msg)
            self._update_status(error_msg, "error")
            raise RuntimeError(error_msg) from e
    
    def _verify_initialization(self) -> None:
        """Verify that all required components are properly initialized.
        
        Raises:
            RuntimeError: If any required component is not properly initialized
        """
        self.logger.debug("Verifying module initialization...")
        
        # Check operation manager
        if not self._operation_manager:
            error_msg = "Operation manager not initialized"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        # Check config handler
        if not self._config_handler:
            error_msg = "Config handler not initialized"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        # Verify required components exist
        required_components = ["setup_button", "header_container", "operation_container"]
        for comp_name in required_components:
            if not self.get_component(comp_name):
                error_msg = f"Required component '{comp_name}' not found"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        # Verify header container has status panel
        header_container = self.get_component('header_container')
        if not hasattr(header_container, 'status_panel'):
            error_msg = "Header container is missing status_panel"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
                
        # Verify required operations are registered
        required_operations = ["full_setup", "status", "reset"]
        for op_name in required_operations:
            if not self.get_operation(op_name):
                error_msg = f"Required operation '{op_name}' not registered"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        self.logger.debug("✅ Module initialization verified successfully")
    
    def _log_initialization_complete(self) -> None:
        """Log initialization completion to operation container."""
        try:
            if self._operation_manager and hasattr(self._operation_manager, 'log'):
                self._operation_manager.log("✅ Colab module initialized successfully", 'info')
                self._operation_manager.log("🔧 Ready for Google Colab environment setup", 'info')
                
                # Log environment detection status
                if self._environment_detected:
                    self._operation_manager.log("🌐 Google Colab environment detected", 'info')
                else:
                    self._operation_manager.log("💻 Local environment detected", 'info')
                
                # Log available operations
                if hasattr(self._operation_manager, 'get_operations'):
                    operations = self._operation_manager.get_operations()
                    self._operation_manager.log(f"📋 Available operations: {', '.join(operations.keys())}", 'info')
                    
            self.logger.debug("✅ Initialization complete logs sent to operation container")
            
        except Exception as e:
            self.logger.warning(f"Failed to log initialization complete: {e}")
    
    def _detect_environment(self) -> None:
        """Detect if running in Google Colab environment."""
        try:
            import google.colab
            self._environment_detected = True
            self.update_config(environment_type="colab", is_colab=True)
            self.logger.info("Google Colab environment detected")
        except ImportError:
            self._environment_detected = False
            self.update_config(environment_type="local", is_colab=False)
            self.logger.info("Local environment detected (not Google Colab)")
    
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
            self.logger.debug("Setting up operation manager...")
            
            # Get operation container instance directly
            operation_container = self.get_component("operation_container")
            if not operation_container:
                self.logger.warning("Operation container not found, creating a new one")
                from smartcash.ui.components.operation_container import OperationContainer
                operation_container = OperationContainer()
            
            # Get current config
            config = self.get_config()
            self.logger.debug(f"Initializing operation manager with config: {config}")
            
            # Initialize operation manager with the operation container
            # The operation container from colab_ui.py is a VBox widget, so pass it directly
            self._operation_manager = ColabOperationManager(
                config=config,
                operation_container=operation_container
            )
            
            # Setup UI logging bridge to capture backend service logs
            self._setup_ui_logging_bridge(operation_container)
            
            # Initialize progress display to show the operation container is working
            self._initialize_progress_display()
            
            # Initialize the operation manager
            self._operation_manager.initialize()
            
            # Log available operations for debugging
            operations = self._operation_manager.get_operations()
            self.logger.debug(f"Operation manager initialized with {len(operations)} operations: {list(operations.keys())}")
            
            # Verify full_setup operation is available
            if 'full_setup' not in operations:
                self.logger.error("❌ full_setup operation not found in operation manager")
                raise ValueError("Required 'full_setup' operation not found in operation manager")
                
            self.logger.info("✅ Operation manager initialized successfully")
            
        except Exception as e:
            error_msg = f"❌ Failed to setup operation manager: {str(e)}"
            self.logger.exception(error_msg)
            self._update_status(error_msg, "error")
            raise
    
    def _setup_config_handler(self) -> None:
        """Setup config handler (no persistence for Colab)."""
        try:
            self._config_handler = ColabConfigHandler()
            self.logger.debug("Config handler initialized (no persistence)")
        except Exception as e:
            self.logger.exception("Failed to setup config handler")
            raise
    
    def _setup_ui_logging_bridge(self, operation_container: Any) -> None:
        """Setup UI logging bridge to capture backend service logs.
        
        This method creates a custom logging handler that captures logs from
        backend services (smartcash.dataset.*, smartcash.model.*, etc.) and
        forwards them to the operation container for display in the UI.
        
        Args:
            operation_container: The operation container instance to log to
        """
        try:
            import logging
            from typing import Callable
            
            # Get the log function from operation container
            log_func = None
            if isinstance(operation_container, dict) and 'log_message' in operation_container:
                log_func = operation_container['log_message']
            elif hasattr(operation_container, 'log_message'):
                log_func = operation_container.log_message
            elif hasattr(self._operation_manager, 'log'):
                log_func = self._operation_manager.log
            
            if not log_func:
                self.logger.warning("⚠️ Could not setup UI logging bridge - log_message function not found")
                return
            
            # Setup basic UI logging for the module
            try:
                from smartcash.ui.core.logging.ui_logging_manager import setup_ui_logging
                setup_ui_logging(
                    module_name='setup.colab',
                    log_message_func=log_func
                )
            except ImportError:
                self.logger.warning("⚠️ UI logging manager not available")
            
            # Create custom handler for backend services
            class BackendUILogHandler(logging.Handler):
                """Custom handler to route backend service logs to UI."""
                
                def __init__(self, log_func: Callable[[str, str], None]):
                    super().__init__()
                    self.log_func = log_func
                    self.setLevel(logging.INFO)  # Capture INFO and above
                    formatter = logging.Formatter('%(name)s: %(message)s')
                    self.setFormatter(formatter)
                    
                def emit(self, record):
                    try:
                        msg = self.format(record)
                        # Map log levels
                        level = 'debug' if record.levelno == logging.DEBUG else \
                               'info' if record.levelno == logging.INFO else \
                               'warning' if record.levelno == logging.WARNING else \
                               'error'
                        
                        # Forward to operation container
                        self.log_func(msg, level)
                    except Exception:
                        pass  # Silently handle logging errors
            
            # Create handler for backend services
            backend_handler = BackendUILogHandler(log_func)
            
            # Configure backend service loggers
            backend_namespaces = [
                'smartcash.dataset',     # Dataset processing services
                'smartcash.model',       # Model services
                'smartcash.common',      # Common services
                'smartcash.setup.colab'  # Colab-specific backend services
            ]
            
            for namespace in backend_namespaces:
                backend_logger = logging.getLogger(namespace)
                
                # Remove existing console handlers to prevent duplicate output
                for handler in backend_logger.handlers[:]:
                    if isinstance(handler, logging.StreamHandler):
                        import sys
                        if hasattr(handler, 'stream') and handler.stream in (sys.stdout, sys.stderr):
                            backend_logger.removeHandler(handler)
                
                # Add UI handler
                backend_logger.addHandler(backend_handler)
                backend_logger.setLevel(logging.INFO)
            
            self.logger.debug("✅ UI logging bridge setup completed for backend services")
            
            # Store handler and namespaces for cleanup
            self._ui_log_handler = backend_handler
            self._ui_backend_loggers = backend_namespaces
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to setup UI logging bridge: {e}")
    
    def _cleanup_ui_logging_bridge(self) -> None:
        """Clean up UI logging bridge handlers."""
        try:
            import logging
            
            # Cleanup UI logging for this module
            try:
                from smartcash.ui.core.logging.ui_logging_manager import cleanup_ui_logging
                cleanup_ui_logging('setup.colab')
            except ImportError:
                pass
            
            # Remove custom handlers from backend services
            if hasattr(self, '_ui_log_handler') and hasattr(self, '_ui_backend_loggers'):
                for namespace in self._ui_backend_loggers:
                    backend_logger = logging.getLogger(namespace)
                    # Remove all handlers that were added by this module
                    for handler in backend_logger.handlers[:]:
                        if hasattr(handler, 'log_func'):  # Our custom handler
                            backend_logger.removeHandler(handler)
                
                self.logger.debug("✅ UI logging bridge cleaned up")
                
                # Clean up references
                delattr(self, '_ui_log_handler')
                delattr(self, '_ui_backend_loggers')
                
        except Exception as e:
            self.logger.debug(f"Error during logging cleanup: {e}")
    
    def _initialize_progress_display(self) -> None:
        """Initialize progress tracker display to show by default."""
        try:
            if not self._operation_manager:
                return
            
            # Initialize progress tracker with default state
            self._operation_manager.update_progress(0, "Ready - No operation running")
            
            # Log initial status to show the operation container is working
            self._operation_manager.log("🚀 Colab module ready", 'info')
            self._operation_manager.log("📋 Progress tracker initialized", 'debug')
            
            self.logger.debug("✅ Progress display initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing progress display: {e}")
    
    def cleanup(self) -> None:
        """Clean up module resources and operation manager."""
        try:
            self.logger.info("🧹 Starting Colab module cleanup...")
            
            # Clean up UI logging bridge
            self._cleanup_ui_logging_bridge()
            
            # Clean up operation manager
            if self._operation_manager:
                self._operation_manager.cleanup()
                self._operation_manager = None
                self.logger.debug("✅ Operation manager cleaned up")
            
            # Clean up config handler
            if self._config_handler:
                # No specific cleanup needed for config handler
                self._config_handler = None
                self.logger.debug("✅ Config handler cleaned up")
            
            # Clean up parent resources
            super().cleanup()
            
            self.logger.info("✅ Colab module cleanup completed")
            
        except Exception as e:
            self.logger.exception(f"❌ Error during cleanup: {e}")
    
    def _register_operations(self) -> None:
        """Register all Colab operations with the operation registry.
        
        This method:
        1. Gets available operations from the operation manager
        2. Registers each operation with the parent UIModule
        3. Registers convenience methods (status, reset)
        4. Ensures required operations exist
        
        Raises:
            RuntimeError: If operation registration fails
        """
        try:
            if not self._operation_manager:
                error_msg = "Operation manager not initialized"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Get operations from manager
            operations = self._operation_manager.get_operations()
            if not operations:
                self.logger.warning("No operations found in operation manager")
            else:
                self.logger.debug(f"Found {len(operations)} operations in operation manager")
            
            # Register each operation from the operation manager
            registered_ops = []
            for op_name, op_func in operations.items():
                try:
                    self.register_operation(op_name, op_func)
                    registered_ops.append(op_name)
                    self.logger.debug(f"✅ Registered operation: {op_name}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to register operation {op_name}: {e}", exc_info=True)
                    raise
            
            # Register convenience methods
            try:
                self.register_operation("status", self.get_environment_status)
                registered_ops.append("status")
                self.logger.debug("✅ Registered convenience operation: status")
                
                self.register_operation("reset", self.reset_environment)
                registered_ops.append("reset")
                self.logger.debug("✅ Registered convenience operation: reset")
            except Exception as e:
                self.logger.error(f"❌ Failed to register convenience operations: {e}", exc_info=True)
                raise
            
            # Ensure required operations exist
            required_operations = ["full_setup"]
            for req_op in required_operations:
                if req_op not in registered_ops:
                    try:
                        if req_op == "full_setup":
                            self.register_operation('full_setup', self.execute_full_setup)
                            registered_ops.append('full_setup')
                            self.logger.warning(f"⚠️ Registered fallback {req_op} operation")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to register required operation {req_op}: {e}")
                        raise RuntimeError(f"Required operation {req_op} not available") from e
            
            self.logger.info(f"✅ Successfully registered {len(registered_ops)} operations: {', '.join(registered_ops)}")
            
        except Exception as e:
            error_msg = f"❌ Failed to register operations: {str(e)}"
            self.logger.exception(error_msg)
            self._update_status(error_msg, "error")
            raise RuntimeError(error_msg) from e
    
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for UI components."""
        try:
            # Connect setup button to handler
            setup_button = self.get_component("setup_button") or self.get_component("primary_button")
            if setup_button:
                # Remove any existing click handlers
                try:
                    # For IPython 8.0+ with CallbackDispatcher
                    if hasattr(setup_button, '_click_handlers') and hasattr(setup_button._click_handlers, 'callbacks'):
                        # Get all callbacks and remove them
                        for callback in list(setup_button._click_handlers.callbacks):
                            setup_button.on_click(callback, remove=True)
                    # For older IPython versions
                    elif hasattr(setup_button, '_click_handlers') and isinstance(setup_button._click_handlers, list):
                        for handler in setup_button._click_handlers[:]:
                            setup_button.on_click(handler, remove=True)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to remove existing click handlers: {str(e)}")
                
                # Add new click handler
                setup_button.on_click(self._handle_setup_button_click)
                
                # Add debug attributes
                setup_button._debug_name = "ColabSetupButton"
                setup_button._debug_handler = self._handle_setup_button_click
                
                # Log the connection
                self.logger.info("✅ Connected setup button to handler")
                print("✅ [DEBUG] Connected setup button to handler")
            else:
                error_msg = "⚠️ No setup button found to connect"
                self.logger.warning(error_msg)
                print(f"[WARNING] {error_msg}")
            
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
            error_msg = f"❌ Failed to setup event handlers: {e}"
            self.logger.error(error_msg)
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
    
    def _handle_setup_button_click(self, button=None) -> None:
        """Handle setup button click event to initiate environment setup.
        
        This method:
        1. Validates required components are available
        2. Sets up the operation environment
        3. Executes the full_setup operation
        4. Handles success/error states with appropriate user feedback
        
        Args:
            button: The button that was clicked (unused, required by ipywidgets)
            
        Returns:
            OperationResult: The result of the setup operation, or None if failed
        """
        # Log button click with debug info
        self.logger.info("🔄 [SETUP] Setup button clicked")
        print("\n" + "="*80)
        print("🔄 [SETUP] Setup button clicked - Starting environment setup...")
        
        # Get operation container for logging
        operation_container = None
        try:
            # Log environment information
            self.logger.info("🔍 [SETUP] Environment information:")
            self.logger.info(f"- Python version: {sys.version}")
            self.logger.info(f"- Working directory: {os.getcwd()}")
            self.logger.info(f"- Environment type: {'Colab' if self.is_colab_environment() else 'Local'}")
            
            # Get operation handler and container
            operation_handler = getattr(self, '_operation_handler', None)
            self.logger.info(f"🔍 [SETUP] Operation handler: {'Found' if operation_handler else 'Not found'}")
            
            if operation_handler:
                operation_container = getattr(operation_handler, '_operation_container', None)
                self.logger.info(f"🔍 [SETUP] Operation container: {'Found' if operation_container else 'Not found'}")
            
            # Log start of operation
            start_msg = "🚀 [SETUP] Starting environment setup..."
            self.logger.info(start_msg)
            print(f"\n{start_msg}")
            self._update_status(start_msg, "info")
            
            if operation_container:
                operation_container.log_message(start_msg, level='info')

            # Get operation manager
            operation_manager = self._operation_manager
            if not operation_manager:
                error_msg = "❌ [ERROR] Operation manager not available"
                self.logger.error(error_msg)
                print(f"\n{error_msg}")
                self._update_status(error_msg, "error")
                if operation_container:
                    operation_container.log_message(error_msg, level='error')
                return None
            
            # Log operation manager info
            self.logger.info(f"🔍 [SETUP] Operation manager: {type(operation_manager).__name__}")
            
            # Get available operations
            available_ops = operation_manager.get_operations()
            self.logger.info(f"🔍 [SETUP] Available operations: {list(available_ops.keys())}")
            print(f"\n🔍 [SETUP] Available operations: {list(available_ops.keys())}")
            
            # Get the full_setup operation function
            full_setup_op = available_ops.get('full_setup')
            if not full_setup_op:
                error_msg = "❌ [ERROR] Full setup operation not found in available operations"
                self.logger.error(error_msg)
                print(f"\n{error_msg}")
                self._update_status(error_msg, "error")
                if operation_container:
                    operation_container.log_message(error_msg, level='error')
                return None
                
            # Log operation function details
            self.logger.info(f"🔍 [SETUP] Full setup operation: {full_setup_op}")
            print(f"\n🔍 [SETUP] Full setup operation: {full_setup_op}")
            
            # Execute the full setup operation
            try:
                self.logger.info("⚙️ [SETUP] Executing full setup operation...")
                print("\n⚙️ [SETUP] Executing full setup operation...")
                
                # Create a progress callback function
                def progress_callback(progress, message=None):
                    progress_msg = f"📊 [PROGRESS] {int(progress*100)}%"
                    if message:
                        progress_msg += f" - {message}"
                    self.logger.info(progress_msg)
                    print(f"\r{progress_msg}", end="", flush=True)
                    
                    # Update status in UI
                    self._update_status(message or f"Progress: {int(progress*100)}%", 
                                     "info" if progress < 1.0 else "success")
                
                # Execute the operation with progress callback
                result = full_setup_op(progress_callback=progress_callback)
                
                # Log completion
                success_msg = "✅ [SUCCESS] Environment setup completed successfully!"
                self.logger.info(success_msg)
                print(f"\n\n{success_msg}")
                self._update_status(success_msg, "success")
                
                return result
                
            except Exception as op_error:
                error_msg = f"❌ [ERROR] Error during setup operation: {str(op_error)}"
                self.logger.error(error_msg, exc_info=True)
                print(f"\n\n{error_msg}")
                self._update_status(f"❌ Error: {str(op_error)}", "error")
                
                # Log full traceback
                import traceback
                tb = traceback.format_exc()
                self.logger.error(f"[TRACEBACK] {tb}")
                print(f"\n[TRACEBACK] {tb}")
                
                return None
            
            # Disable all buttons during operation
            button_states = self._get_operation_manager().disable_all_buttons("⏳ Setting up...")
            
            try:
                self.logger.info("🔧 Executing full environment setup...")
                
                # Execute the operation with progress tracking
                result = self.execute_operation(
                    "full_setup",
                    full_setup_op,
                    message="Performing full environment setup..."
                )
                
                # Handle operation result
                self._handle_setup_result(result, operation_container)
                return result
                
            except Exception as e:
                error_msg = f"❌ Unexpected error during setup: {str(e)}"
                self.logger.exception(error_msg)
                self._update_status(error_msg, "error")
                if operation_container:
                    operation_container.log_message(error_msg, level='error')
                raise
                
            finally:
                # Always restore button states
                if button_states:
                    self._get_operation_manager().restore_button_states(button_states)
        
        except Exception as e:
            error_msg = f"❌ Setup failed: {str(e)}"
            self.logger.exception("Error in setup button handler")
            self._update_status(error_msg, "error")
            
            # Log to operation container if available
            if operation_container:
                operation_container.log_message(error_msg, level='error')
                
            # Restore buttons in error state
            if button_states:
                self._get_operation_manager().enable_all_buttons(
                    button_states, 
                    success=False,
                    error_message="❌ Error"
                )
                
            return None
    
    def get_operation(self, operation_name: str) -> Optional[Callable]:
        """Get an operation by name.
        
        Args:
            operation_name: Name of the operation to retrieve
            
        Returns:
            Callable operation function if found, None otherwise
        """
        if not hasattr(self, '_operations') or not self._operations:
            return None
        return self._operations.get(operation_name)
    
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
    
    def _handle_setup_result(self, result: Any, operation_container: Any = None) -> None:
        """Handle the result of a setup operation.
        
        Args:
            result: The operation result object
            operation_container: Optional operation container for logging
        """
        try:
            if not result:
                error_msg = "❌ Setup failed: No result returned"
                self.logger.error(error_msg)
                self._update_status(error_msg, "error")
                if operation_container:
                    operation_container.log_message(error_msg, level='error')
                return
                
            # Handle operation result (result is an OperationResult object)
            if hasattr(result, 'status') and result.status == OperationStatus.COMPLETED:
                status_msg = "✅ Environment setup completed successfully!"
                self._update_status(status_msg, "success")
                self.logger.info(status_msg)
                
                if operation_container:
                    operation_container.log_message(status_msg, level='success')
            else:
                error_msg = getattr(result, 'message', "Setup failed")
                if hasattr(result, 'error') and result.error:
                    error_msg = f"{error_msg}: {str(result.error)}"
                status_msg = f"❌ {error_msg}"
                self._update_status(status_msg, "error")
                self.logger.error(status_msg)
                
                if operation_container:
                    operation_container.log_message(status_msg, level='error')
                    
        except Exception as e:
            error_msg = f"❌ Error handling setup result: {str(e)}"
            self.logger.exception(error_msg)
            self._update_status(error_msg, "error")
            if operation_container:
                operation_container.log_message(error_msg, level='error')
    
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

def initialize_colab_ui(
    config: Optional[Dict[str, Any]] = None,
    display: bool = True,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Initialize and display the Colab UI.
    
    Args:
        config: Optional configuration dictionary
        display: Whether to display the UI immediately
        **kwargs: Additional arguments
        
    Returns:
        If display=True: Returns None (displays UI directly)
        If display=False: Returns a dictionary with UI components and status
    """
    try:
        # Get the module and UI components
        module = get_colab_uimodule(config=config, **kwargs)
        ui_components = module.list_components()
        
        # Prepare the result dictionary
        result = {
            'success': True,
            'module': module,
            'ui_components': {comp: module.get_component(comp) for comp in ui_components},
            'status': module.get_environment_status()
        }
        
        # Display the UI if requested
        if display and ui_components:
            from IPython import get_ipython
            from IPython.display import display as ipython_display, clear_output
            
            # Clear any existing output
            if get_ipython() is not None:
                clear_output(wait=True)
            
            # Get the main UI container and display it
            main_ui = module.get_component('main_container')
            if main_ui is not None:
                try:
                    # Display the widget directly
                    ipython_display(main_ui)
                except Exception as e:
                    # Fallback to simple display if anything goes wrong
                    logger = get_module_logger("smartcash.ui.setup.colab")
                    logger.error(f"Error displaying UI: {str(e)}")
                    ipython_display(main_ui)
                return None  # Don't return data when display=True
        
        return result
        
    except Exception as e:
        # Always return a dictionary, even on error
        return {
            'success': False,
            'error': str(e),
            'module': None,
            'ui_components': {},
            'status': {}
        }

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
    """Display Colab UI using new UIModule pattern.
    
    This function initializes and displays the Colab UI with the provided configuration.
    It includes proper logging for debugging and error tracking.
    
    Args:
        config: Optional configuration dictionary for the Colab UI
    """
    # Get logger for this function
    logger = get_module_logger("smartcash.ui.setup.colab.display")
    
    try:
        logger.info("🚀 Starting Colab UI display...")
        print("[DEBUG] Starting Colab UI display...")  # Fallback log
        
        # Initialize the UI with the provided config
        logger.debug(f"Initializing UI with config: {config}")
        ui_module = initialize_colab_ui(config)
        
        if ui_module is None:
            error_msg = "❌ Failed to initialize Colab UI: UI module is None"
            logger.error(error_msg)
            print(f"[ERROR] {error_msg}")
            return
            
        logger.info("✅ Colab UI initialized successfully")
        print("[DEBUG] Colab UI initialized successfully")  # Fallback log
        
        # Display the main container
        main_container = ui_module.get_component('main_container')
        if main_container:
            display(main_container)
            logger.info("🎉 Colab UI displayed successfully")
            print("[DEBUG] Colab UI displayed successfully")  # Fallback log
        else:
            error_msg = "❌ Failed to get main container from UI module"
            logger.error(error_msg)
            print(f"[ERROR] {error_msg}")
            raise ValueError("Main container not found in UI module")
        
    except Exception as e:
        error_msg = f"❌ Error displaying Colab UI: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        raise

# Note: Template and shared methods are registered on-demand in create_colab_uimodule()
# to avoid logs during import