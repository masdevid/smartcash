"""
File: smartcash/ui/dataset/downloader/downloader_uimodule.py
Description: Dataset Downloader UIModule implementation using new core UIModule pattern.

This module refactors the dataset downloader to use the new UIModule architecture
while preserving its unique form, backend integration flow, and existing functionality.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime

# Core UI imports
from smartcash.ui.core.ui_module import UIModule, register_operation_method, SharedMethodRegistry
from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.utils.log_suppression import suppress_ui_init_logs

# Downloader module imports
from smartcash.ui.dataset.downloader.components.downloader_ui import create_downloader_ui
from smartcash.ui.dataset.downloader.configs.downloader_config_handler import (
    DownloaderConfigHandler
)
from smartcash.ui.dataset.downloader.configs.downloader_defaults import (
    get_default_downloader_config
)
from smartcash.ui.dataset.downloader.operations.manager import DownloaderOperationManager
from smartcash.ui.dataset.downloader.services.downloader_service import DownloaderService

# Global module instance for singleton pattern
_downloader_uimodule: Optional[UIModule] = None

def register_downloader_template() -> None:
    """Register Downloader module template with UIModuleFactory."""
    from smartcash.ui.core.ui_module_factory import UIModuleFactory, ModuleTemplate
    
    # Create a template for the downloader module
    template = ModuleTemplate(
        module_name="downloader",
        parent_module="dataset",
        default_config=get_default_downloader_config(),
        required_components=[
            "main_container", "header_container", "form_container", 
            "action_container", "operation_container", "footer_container"
        ],
        required_operations=[
            "download", "check", "cleanup", "validate_config", "get_dataset_count"
        ],
        auto_initialize=False,
        description="Dataset downloader module with backend integration"
    )
    
    try:
        UIModuleFactory.register_template(template, overwrite=True)
        local_logger = get_module_logger("smartcash.ui.dataset.downloader.template")
        local_logger.debug("📋 Registered Downloader template")
    except Exception as e:
        local_logger = get_module_logger("smartcash.ui.dataset.downloader.template")
        local_logger.error(f"❌ Failed to register template: {e}")

def register_downloader_shared_methods() -> None:
    """Register Downloader-specific shared methods."""
    
    def validate_roboflow_config(workspace: str, project: str, version: str, api_key: str) -> Dict[str, Any]:
        """Validate Roboflow configuration parameters."""
        try:
            from smartcash.ui.dataset.downloader.services import get_config_validator
            validator = get_config_validator()
            config = {
                'data': {
                    'roboflow': {
                        'workspace': workspace,
                        'project': project,
                        'version': version,
                        'api_key': api_key
                    }
                }
            }
            return validator.validate_config(config)
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def get_existing_dataset_count() -> Dict[str, Any]:
        """Get count of existing dataset files."""
        try:
            from smartcash.ui.dataset.downloader.services import get_dataset_scanner
            scanner = get_dataset_scanner()
            count = scanner.get_existing_dataset_count()
            return {"success": True, "count": count}
        except Exception as e:
            return {"success": False, "error": str(e), "count": 0}
    
    def get_api_key_from_secrets() -> Dict[str, Any]:
        """Get API key from Colab secrets."""
        try:
            from smartcash.ui.dataset.downloader.services import get_secret_manager
            secret_manager = get_secret_manager()
            api_key = secret_manager.get_api_key()
            return {"success": True, "api_key": api_key}
        except Exception as e:
            return {"success": False, "error": str(e), "api_key": None}
    
    # Register methods with error handling for re-registration
    methods = [
        ("validate_roboflow_config", validate_roboflow_config, "Validate Roboflow configuration"),
        ("get_existing_dataset_count", get_existing_dataset_count, "Get existing dataset count"),
        ("get_api_key_from_secrets", get_api_key_from_secrets, "Get API key from secrets")
    ]
    
    for name, method, desc in methods:
        try:
            register_operation_method(name, method, description=desc)
        except ValueError:
            SharedMethodRegistry.register_method(name, method, overwrite=True, 
                                               description=desc, category="operations")
    
    local_logger = get_module_logger("smartcash.ui.dataset.downloader.methods")
    local_logger.debug("🔗 Registered Downloader shared methods")

class DownloaderUIModule(UIModule):
    """
    Dataset Downloader UIModule following the backbone module pattern.
    
    Features:
    - 📥 Dataset download operations
    - 🔍 Dataset verification and validation
    - 🧹 Cleanup operations
    - 📊 Progress tracking and logging integration
    - 🛡️ Error handling with user feedback
    - 🎯 Button management with disable/enable functionality
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Downloader UIModule.
        
        Args:
            config: Downloader configuration (optional, uses defaults if not provided)
        """
        # Initialize instance variables first
        self._ui_components = {}
        self._operation_manager = None
        self._downloader_service = None
        self._config_handler = None
        
        # Initialize with module metadata
        super().__init__(
            module_name='downloader',
            parent_module='dataset',
            config=config or {},
            auto_initialize=False
        )
        
        # Initialize logger
        self.logger = get_module_logger("smartcash.ui.dataset.downloader")
        
        # Initialize configuration handler
        self._initialize_config_handler()
        
    def log(self, message: str, level: str = 'info') -> None:
        """Log a message with the specified log level.
        
        Args:
            message: The message to log
            level: Log level ('debug', 'info', 'warning', 'error', 'critical')
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
        
        self.logger.debug("✅ DownloaderUIModule initialized")
        
    @suppress_ui_init_logs(duration=3.0)
    def initialize(self, config: Dict[str, Any] = None) -> 'DownloaderUIModule':
        """Initialize Downloader module with backend service integration.
        
        Args:
            config: Additional configuration to merge
            
        Returns:
            Self for method chaining
            
        Raises:
            RuntimeError: If initialization fails
        """
        self.logger.info("🔄 Initializing Downloader module...")
        
        try:
            # Update config if provided
            if config:
                self.logger.debug("Merging provided configuration")
                self.update_config(config)
            
            # Initialize components in proper order
            self._ui_components = self._create_ui_components()
            self._initialize_config_handler()
            self._setup_operation_manager()
            self._setup_downloader_service()
            
            # Register operations and set up handlers
            self._register_operations()
            self._setup_event_handlers()
            
            # Initialize parent
            super().initialize()
            
            # Log initialization completion to operation container
            self._log_initialization_complete()
            
            # Mark as initialized
            self._is_initialized = True
            self._initialization_error = None
            
            self.logger.info("✅ Downloader module initialized successfully")
            return self
            
        except Exception as e:
            error_msg = f"Failed to initialize Downloader module: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._initialization_error = str(e)
            self._is_initialized = False
            self._update_status(f"Initialization failed: {str(e)}", "error")
            raise RuntimeError(error_msg) from e
    
    def _log_initialization_complete(self) -> None:
        """Log initialization completion to operation container."""
        try:
            if self._operation_manager and hasattr(self._operation_manager, 'log'):
                self._operation_manager.log("✅ Downloader module initialized successfully", 'info')
                self._operation_manager.log("📥 Ready for dataset download operations", 'info')
                
                # Log available operations
                if hasattr(self._operation_manager, 'get_operations'):
                    operations = self._operation_manager.get_operations()
                    self._operation_manager.log(f"📋 Available operations: {', '.join(operations.keys())}", 'info')
                    
            self.logger.debug("✅ Initialization complete logs sent to operation container")
            
        except Exception as e:
            self.logger.warning(f"Failed to log initialization complete: {e}")
    
    def get_main_widget(self):
        """Get the main widget for display."""
        try:
            # First try to get the main container from registered components
            main_widget = self.get_component('main_container')
            
            # If not found, try alternative names
            if main_widget is None:
                main_widget = self.get_component('ui')
                
                # If still not found, try to get it from the _ui_components dictionary
                if main_widget is None and hasattr(self, '_ui_components'):
                    if 'main_container' in self._ui_components:
                        main_widget = self._ui_components['main_container']
                    elif 'ui' in self._ui_components:
                        main_widget = self._ui_components['ui']
            
            # If we have a main widget but it has a container attribute, use that
            if hasattr(main_widget, 'container'):
                main_widget = main_widget.container
                
            return main_widget
            
        except Exception as e:
            self.logger.error(f"❌ Error getting main widget: {e}")
            return None
    
    def _create_ui_components(self) -> Dict[str, Any]:
        """Create and register UI components."""
        try:
            self.logger.info("🛠️ Creating UI components...")
            
            # Initialize UI components dictionary
            ui_components = {}
            registered_components = set()
            
            # 1. Create operation container first (needed for logging during other component creation)
            from smartcash.ui.components.operation_container import OperationContainer
            operation_container = OperationContainer(
                title="Download Operations",
                show_header=True,
                collapsible=True,
                collapsed=False
            )
            
            if operation_container:
                ui_components['operation_container'] = operation_container
                self.register_component('operation_container', operation_container)
                registered_components.add('operation_container')
                self.logger.info("✅ Created operation container")
            
            # 2. Create the full downloader UI to get the action container and buttons
            from smartcash.ui.dataset.downloader.components.downloader_ui import create_downloader_ui
            
            # Create the downloader UI which includes the action container
            downloader_ui = create_downloader_ui(config=self._config)
            
            # Extract the action container and buttons from the UI
            action_container = downloader_ui.get('action_container')
            if action_container:
                ui_components['action_container'] = action_container
                self.register_component('action_container', action_container)
                registered_components.add('action_container')
                self.logger.debug("✅ Created action container with buttons")
                
                # Register individual buttons
                for btn_id in ['download_button', 'check_button', 'cleanup_button']:
                    if btn_id in downloader_ui and downloader_ui[btn_id] is not None:
                        ui_components[btn_id] = downloader_ui[btn_id]
                        self.register_component(btn_id, downloader_ui[btn_id])
                        registered_components.add(btn_id)
                        self.logger.debug(f"✅ Registered button: {btn_id}")
                    else:
                        self.logger.warning(f"⚠️ Button not found in downloader UI: {btn_id}")
            
            # 3. Extract and register any additional components from downloader_ui
            for comp_name, comp in downloader_ui.items():
                if comp_name not in ui_components and comp is not None:
                    ui_components[comp_name] = comp
                    self.register_component(comp_name, comp)
                    registered_components.add(comp_name)
                    self.logger.debug(f"✅ Registered component: {comp_name}")
            
            # 4. Verify all required buttons are registered and valid
            button_references = {}
            for btn_id in ['download_button', 'check_button', 'cleanup_button']:
                btn = self.get_component(btn_id)
                if btn is not None and hasattr(btn, 'on_click'):
                    button_references[btn_id] = btn
                    setattr(self, f'_{btn_id}', btn)
                    self.logger.info(f"✅ Verified button: {btn_id} ({type(btn).__name__})")
                else:
                    self.logger.error(f"❌ Button not found or invalid: {btn_id}")
            
            # 5. Log final status
            if len(button_references) == 3:
                self.logger.info("🎉 Successfully registered all buttons!")
            else:
                self.logger.warning(f"⚠️ Only {len(button_references)} out of 3 buttons were registered")
            
            self.logger.info(f"📊 Total registered components: {len(registered_components)}")
            
            # 6. Store the UI components for use by other methods
            self._ui_components = ui_components
            
            # 9. Return the UI components for use by other methods
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create UI components: {e}", exc_info=True)
            raise
    
    def _initialize_config_handler(self) -> None:
        """Initialize the configuration handler with defaults and UI components.
        
        This method sets up the configuration handler with default values and any
        provided configuration, then initializes it with the current UI components.
        
        Raises:
            RuntimeError: If configuration handler initialization fails
        """
        try:
            self.logger.debug("🔄 Initializing configuration handler...")
            
            # Get default config
            default_config = get_default_downloader_config()
            
            # Create config handler with merged config
            self._config_handler = DownloaderConfigHandler(
                module_name='downloader',
                parent_module='dataset',
                ui_components=self._ui_components or {},
                config=self._config or {},
                use_shared_config=True
            )
            
            # Load and validate config
            config = self._config_handler.load_config()
            self._config = self._config_handler.merge_config(default_config, config)
            
            # Update config in handler
            self._config_handler.update_config(self._config)
            
            self.logger.debug("✅ Configuration handler initialized")
            
        except Exception as e:
            error_msg = f"❌ Failed to initialize configuration handler: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.log(error_msg, 'error')
            raise RuntimeError(error_msg) from e

    def _setup_downloader_service(self) -> None:
        """Setup downloader service for backend integration.
        
        This method initializes the DownloaderService instance that handles
        communication with the backend downloader functionality.
        
        Raises:
            RuntimeError: If downloader service initialization fails
        """
        try:
            self._downloader_service = DownloaderService(logger=self.logger)
            self.logger.debug("🔧 Setup downloader service")
            
        except Exception as e:
            error_msg = f"❌ Failed to setup downloader service: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
            
    def _setup_operation_manager(self) -> None:
        """Setup the operation manager for handling download operations.
        
        This method initializes the DownloaderOperationManager which handles
        the execution of download, check, and cleanup operations.
        
        Raises:
            RuntimeError: If operation manager initialization fails
        """
        if not hasattr(self, '_ui_components') or not self._ui_components:
            error_msg = "UI components not available for operation manager"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        try:
            self.logger.info("Initializing operation manager...")
            
            operation_container = self._ui_components.get('operation_container')
            if not operation_container:
                raise ValueError("Operation container not found in UI components")
            
            self._operation_manager = DownloaderOperationManager(
                config=self.get_config(),
                operation_container=operation_container
            )
            
            self.logger.info("✅ Operation manager initialized")
            
        except Exception as e:
            error_msg = f"Failed to initialize operation manager: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def _register_operations(self) -> None:
        """Register downloader operations with the operation manager."""
        try:
            if not self._operation_manager:
                raise ValueError("Operation manager not initialized")
            
            # Get operations from manager
            operations = self._operation_manager.get_operations()
            
            # Register each operation
            for op_name, op_func in operations.items():
                self.register_operation(op_name, op_func)
            
            # Register additional convenience methods
            self.register_operation("status", self.get_downloader_status)
            self.register_operation("reset", self.reset_downloader)
            self.register_operation("validate_config", self.validate_configuration)
            self.register_operation("get_dataset_count", self.get_existing_dataset_count)
            
            self.logger.debug(f"⚙️ Registered {len(operations)} operations")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register operations: {e}")
            raise
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for UI components."""
        try:
            # Get all button references
            buttons = {
                'download': getattr(self, '_download_button', None) or self.get_component('download_button'),
                'check': getattr(self, '_check_button', None) or self.get_component('check_button'),
                'cleanup': getattr(self, '_cleanup_button', None) or self.get_component('cleanup_button')
            }
            
            # Connect all buttons
            connected_buttons = []
            
            for btn_type, button in buttons.items():
                if button is None:
                    self.logger.error(f"❌ No {btn_type} button found to connect")
                    continue
                    
                # Get the appropriate handler method
                handler = getattr(self, f'_handle_{btn_type}_button_click', None)
                if handler is None:
                    self.logger.error(f"❌ No handler found for {btn_type} button")
                    continue
                
                try:
                    # Create a new callback handler with just our handler
                    # This is the safest way to ensure we don't interfere with ipywidgets internals
                    try:
                        # First, remove any existing click handlers
                        button.on_click(lambda _: None, remove=True)
                    except Exception as e:
                        # If removing fails, we'll continue anyway
                        self.logger.debug(f"⚠️ Could not remove existing click handlers: {str(e)}")
                    
                    try:
                        # Add our new handler
                        button.on_click(handler)
                        
                        # If we got here, assume it worked
                        connected_buttons.append(btn_type)
                        self.logger.info(f"✅ Connected {btn_type} button to handler (type: {type(button).__name__})")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to connect {btn_type} button: {str(e)}")
                        raise
                        
                except Exception as e:
                    self.logger.error(f"❌ Error connecting {btn_type} button: {str(e)}", exc_info=True)
            
            # Log connection summary
            if connected_buttons:
                self.logger.info(f"🔌 Successfully connected buttons: {', '.join(connected_buttons)}")
            else:
                self.logger.error("❌ No buttons were connected successfully")
                
            # If no buttons were connected, try an alternative approach
            if not connected_buttons:
                self.logger.warning("⚠️ Trying alternative approach to connect buttons...")
                self._connect_buttons_alternative()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to setup event handlers: {e}", exc_info=True)
            
    def _connect_buttons_alternative(self) -> None:
        """Alternative method to connect buttons if the standard method fails."""
        try:
            # Try to get buttons directly from the action container
            action_container = getattr(self, '_action_container', None) or self.get_component('action_container')
            if not action_container:
                self.logger.error("❌ Action container not found")
                return
                
            # Get buttons from action container
            buttons = {}
            if hasattr(action_container, 'get') and callable(action_container.get):
                buttons = action_container.get('buttons', {})
            
            # Connect each button
            connected = 0
            for btn_type in ['download', 'check', 'cleanup']:
                btn_id = f"{btn_type}_button"
                button = buttons.get(btn_id)
                
                if button is None:
                    self.logger.warning(f"⚠️ {btn_id} not found in action container")
                    continue
                    
                # Get the handler method
                handler = getattr(self, f'_handle_{btn_type}_button_click', None)
                if handler is None:
                    self.logger.warning(f"⚠️ No handler found for {btn_id}")
                    continue
                    
                try:
                    # Clear existing handlers
                    if hasattr(button, '_click_handlers'):
                        button._click_handlers.clear()
                        
                    # Connect the handler
                    button.on_click(handler)
                    
                    # Store direct reference
                    setattr(self, f'_{btn_id}', button)
                    self.register_component(btn_id, button)
                    
                    connected += 1
                    self.logger.info(f"🔌 Successfully connected {btn_id} (alternative method)")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to connect {btn_id}: {str(e)}")
            
            if connected > 0:
                self.logger.info(f"✅ Successfully connected {connected} buttons using alternative method")
            else:
                self.logger.error("❌ Failed to connect any buttons using alternative method")
                
        except Exception as e:
            self.logger.error(f"❌ Error in alternative button connection: {str(e)}", exc_info=True)
    
    def _handle_download_button_click(self, button=None) -> None:
        """Handle download button click event."""
        try:
            self.logger.info("📥 Download button clicked - starting dataset download")
            self._update_status("Starting dataset download...", "info")
            
            # Get UI configuration from form inputs
            ui_config = self._extract_ui_config()
            
            # Execute download operation
            result = self.execute_download(ui_config)
            
            # Update status based on result
            success = result.get("success", False)
            if success:
                self._update_status("Dataset download completed successfully!", "success")
                self.logger.info("✅ Dataset download completed successfully!")
            else:
                error_msg = result.get("error", "Download failed")
                self._update_status(f"Download failed: {error_msg}", "error")
                self.logger.error(f"❌ Download failed: {error_msg}")
            
            self.logger.debug(f"Download completed with result: {result}")
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"❌ Download button click failed: {error_msg}", exc_info=True)
            self._update_status(f"Download error: {error_msg}", "error")
    
    def _handle_check_button_click(self, button=None) -> None:
        """Handle check button click event."""
        try:
            self.logger.info("🔍 Check button clicked - checking dataset status")
            self._update_status("Checking dataset status...", "info")
            
            # Execute check operation
            result = self.execute_check()
            
            # Update status based on result
            success = result.get("success", False)
            if success:
                count = result.get("count", 0)
                self._update_status(f"Dataset check completed - {count} files found", "success")
                self.logger.info(f"✅ Dataset check completed - {count} files found")
            else:
                error_msg = result.get("error", "Check failed")
                self._update_status(f"Check failed: {error_msg}", "error")
                self.logger.error(f"❌ Check failed: {error_msg}")
        
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"❌ Check button click failed: {error_msg}", exc_info=True)
            self._update_status(f"Check error: {error_msg}", "error")
    
    def _handle_cleanup_button_click(self, button=None) -> None:
        """Handle cleanup button click event."""
        try:
            self.logger.info("🧹 Cleanup button clicked - starting dataset cleanup")
            self._update_status("Starting dataset cleanup...", "info")
            
            # Execute cleanup operation
            result = self.execute_cleanup()
            
            # Update status based on result
            success = result.get("success", False)
            if success:
                cleaned_count = result.get("cleaned_count", 0)
                self._update_status(f"Cleanup completed - {cleaned_count} items cleaned", "success")
                self.logger.info(f"✅ Cleanup completed - {cleaned_count} items cleaned")
            else:
                error_msg = result.get("error", "Cleanup failed")
                self._update_status(f"Cleanup failed: {error_msg}", "error")
                self.logger.error(f"❌ Cleanup failed: {error_msg}")
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"❌ Cleanup button click failed: {error_msg}", exc_info=True)
            self._update_status(f"Cleanup error: {error_msg}", "error")
    
    def _extract_ui_config(self) -> Dict[str, Any]:
        """Extract configuration from UI form inputs."""
        try:
            form_widgets = self.get_component("form_widgets")
            if form_widgets is None:
                form_widgets = {}
            
            # Extract values from form inputs
            ui_config = {
                'workspace': getattr(form_widgets.get('workspace_input'), 'value', ''),
                'project': getattr(form_widgets.get('project_input'), 'value', ''),
                'version': getattr(form_widgets.get('version_input'), 'value', ''),
                'api_key': getattr(form_widgets.get('api_key_input'), 'value', ''),
                'validate_download': getattr(form_widgets.get('validate_checkbox'), 'value', True),
                'backup_existing': getattr(form_widgets.get('backup_checkbox'), 'value', False),
            }
            
            return ui_config
            
        except Exception as e:
            self.logger.error(f"Error extracting UI config: {e}", exc_info=True)
            return {}
    
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
    
    def get_downloader_status(self) -> Dict[str, Any]:
        """Get comprehensive downloader status.
        
        Returns:
            Dictionary with downloader information
        """
        status = {
            "module": self.full_module_name,
            "module_status": self.get_status().value,
            "ready": self.is_ready(),
            "error_count": getattr(self, '_error_count', 0),
            "components": len(self.list_components()),
            "operations": len(self.list_operations()),
            "backend_service": self._downloader_service is not None,
            "config_handler": self._config_handler is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add dataset information if service is available
        if self._downloader_service:
            try:
                status["existing_dataset_count"] = self._downloader_service.get_existing_dataset_count()
            except Exception as e:
                status["existing_dataset_count"] = 0
                status["dataset_count_error"] = str(e)
        
        return status
    
    def reset_downloader(self) -> Dict[str, Any]:
        """Reset downloader to initial state.
        
        Returns:
            Dictionary with reset results
        """
        try:
            # Reset components
            self.clear_components()
            
            # Reset operation manager
            if self._operation_manager:
                # Clean up operation manager if it has cleanup method
                if hasattr(self._operation_manager, 'cleanup'):
                    self._operation_manager.cleanup()
            
            # Re-create components
            self._create_ui_components()
            
            # Re-setup services and managers
            self._setup_downloader_service()
            self._setup_operation_manager()
            self._setup_config_handler()
            
            # Re-register operations
            self._register_operations()
            
            # Re-setup event handlers
            self._setup_event_handlers()
            
            return {
                "success": True,
                "message": "Downloader reset successfully",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to reset downloader: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to reset downloader"
            }
    
    def execute_download(self, ui_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute dataset download workflow.
        
        Args:
            ui_config: UI configuration from form inputs
            
        Returns:
            Dictionary with download results
        """
        if not self._operation_manager:
            return {
                "success": False,
                "error": "Operation manager not initialized",
                "message": "Cannot execute download without operation manager"
            }
        
        try:
            if ui_config is None:
                ui_config = self._extract_ui_config()
            
            # Execute download operation
            import asyncio
            
            # Check if we're in a Jupyter notebook environment with an existing event loop
            try:
                from IPython import get_ipython
                
                if get_ipython() is not None:
                    # In Jupyter, schedule the coroutine to run in the existing event loop
                    future = asyncio.ensure_future(self._operation_manager.execute_download(ui_config))
                    # Return a placeholder result, the actual result will be handled by the event loop
                    return {
                        "success": True,
                        "message": "Download operation started in background"
                    }
            except (ImportError, AttributeError):
                # Not in Jupyter, handle normally
                pass
                
            # Fall back to synchronous execution if not in Jupyter or if there was an error
            if asyncio.iscoroutinefunction(self._operation_manager.execute_download):
                # If there's no running event loop, create one
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                if loop.is_running():
                    # If loop is already running, schedule the task
                    future = asyncio.ensure_future(self._operation_manager.execute_download(ui_config))
                    return {
                        "success": True,
                        "message": "Download operation started in background"
                    }
                else:
                    # Otherwise, run the coroutine directly
                    return loop.run_until_complete(self._operation_manager.execute_download(ui_config))
            else:
                # Not a coroutine, just call it directly
                return self._operation_manager.execute_download(ui_config)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to execute download: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to execute dataset download"
            }
    
    def execute_check(self) -> Dict[str, Any]:
        """Execute dataset check workflow.
        
        Returns:
            Dictionary with check results
        """
        if not self._operation_manager:
            return {
                "success": False,
                "error": "Operation manager not initialized"
            }
        
        try:
            # Import asyncio only when needed
            import asyncio
            
            # Check if we're in a Jupyter notebook environment with an existing event loop
            try:
                import ipykernel
                from IPython import get_ipython
                
                if get_ipython() is not None:
                    # In Jupyter, schedule the coroutine to run in the existing event loop
                    future = asyncio.ensure_future(self._operation_manager.execute_check())
                    # Return a placeholder result, the actual result will be handled by the event loop
                    return {
                        "success": True,
                        "message": "Check operation started in background"
                    }
            except (ImportError, AttributeError):
                # Not in Jupyter, handle normally
                pass
                
            # Fall back to synchronous execution if not in Jupyter or if there was an error
            if asyncio.iscoroutinefunction(self._operation_manager.execute_check):
                # If there's no running event loop, create one
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                if loop.is_running():
                    # If loop is already running, schedule the task
                    future = asyncio.ensure_future(self._operation_manager.execute_check())
                    return {
                        "success": True,
                        "message": "Check operation started in background"
                    }
                else:
                    # Otherwise, run the coroutine directly
                    return loop.run_until_complete(self._operation_manager.execute_check())
            else:
                # Not a coroutine, just call it directly
                return self._operation_manager.execute_check()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to execute check: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def execute_cleanup(self, targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute dataset cleanup workflow.
        
        Args:
            targets: Optional list of cleanup targets
            
        Returns:
            Dictionary with cleanup results
        """
        if not self._operation_manager:
            return {
                "success": False,
                "error": "Operation manager not initialized"
            }
        
        try:
            # Import asyncio only when needed
            import asyncio
            
            # Check if we're in a Jupyter notebook environment with an existing event loop
            try:
                from IPython import get_ipython
                
                if get_ipython() is not None:
                    # In Jupyter, schedule the coroutine to run in the existing event loop
                    future = asyncio.ensure_future(self._operation_manager.execute_cleanup(targets))
                    # Return a placeholder result, the actual result will be handled by the event loop
                    return {
                        "success": True,
                        "message": "Cleanup operation started in background"
                    }
            except (ImportError, AttributeError):
                # Not in Jupyter, handle normally
                pass
                
            # Fall back to synchronous execution if not in Jupyter or if there was an error
            if asyncio.iscoroutinefunction(self._operation_manager.execute_cleanup):
                # If there's no running event loop, create one
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                if loop.is_running():
                    # If loop is already running, schedule the task
                    future = asyncio.ensure_future(self._operation_manager.execute_cleanup(targets))
                    return {
                        "success": True,
                        "message": "Cleanup operation started in background"
                    }
                else:
                    # Otherwise, run the coroutine directly
                    return loop.run_until_complete(self._operation_manager.execute_cleanup(targets))
            else:
                # Not a coroutine, just call it directly
                return self._operation_manager.execute_cleanup(targets)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to execute cleanup: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_configuration(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate downloader configuration.
        
        Args:
            config: Configuration to validate (uses current config if not provided)
            
        Returns:
            Dictionary with validation results
        """
        if not self._downloader_service:
            return {
                "valid": False,
                "error": "Downloader service not initialized"
            }
        
        try:
            if config is None:
                config = self.get_config()
            
            return self._downloader_service.validate_config(config)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate configuration: {e}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    def get_existing_dataset_count(self) -> Dict[str, Any]:
        """Get count of existing dataset files.
        
        Returns:
            Dictionary with dataset count information
        """
        if not self._downloader_service:
            return {
                "success": False,
                "count": 0,
                "error": "Downloader service not initialized"
            }
        
        try:
            count = self._downloader_service.get_existing_dataset_count()
            return {
                "success": True,
                "count": count
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get dataset count: {e}")
            return {
                "success": False,
                "count": 0,
                "error": str(e)
            }
    
    def get_operation_manager(self) -> Optional[DownloaderOperationManager]:
        """Get the operation manager instance.
        
        Returns:
            DownloaderOperationManager instance or None
        """
        return self._operation_manager
    
    def get_config_handler(self) -> Optional[DownloaderConfigHandler]:
        """Get the config handler instance.
        
        Returns:
            DownloaderConfigHandler instance or None
        """
        return self._config_handler
    
    def get_downloader_service(self) -> Optional[DownloaderService]:
        """Get the downloader service instance.
        
        Returns:
            DownloaderService instance or None
        """
        return self._downloader_service
    
    def get_ui_components(self) -> Dict[str, Any]:
        """Get UI components dictionary.
        
        Returns:
            UI components dictionary
        """
        return self._ui_components or {}

def create_downloader_uimodule(config: Dict[str, Any] = None, 
                              auto_initialize: bool = True,
                              force_new: bool = False) -> DownloaderUIModule:
    """Create Downloader UIModule using factory pattern.
    
    Args:
        config: Downloader configuration dictionary
        auto_initialize: Whether to auto-initialize the module
        force_new: Force creation of new instance
        
    Returns:
        DownloaderUIModule instance
    """
    global _downloader_uimodule
    
    # Return existing instance if available and not forcing new
    if not force_new and _downloader_uimodule is not None:
        if config:
            _downloader_uimodule.update_config(**config)
        return _downloader_uimodule
    
    try:
        # Ensure template is registered
        register_downloader_template()
        
        # Ensure shared methods are registered
        register_downloader_shared_methods()
        
        # Create new module instance
        module = DownloaderUIModule(config)
        
        # Initialize if requested
        if auto_initialize:
            module.initialize()
        
        # Store global reference
        _downloader_uimodule = module
        
        local_logger = get_module_logger("smartcash.ui.dataset.downloader.factory")
        local_logger.debug(f"🏭 Created Downloader UIModule")
        return module
        
    except Exception as e:
        local_logger = get_module_logger("smartcash.ui.dataset.downloader.factory")
        local_logger.error(f"❌ Failed to create Downloader UIModule: {e}")
        raise

def get_downloader_uimodule(create_if_missing: bool = True, **kwargs) -> Optional[DownloaderUIModule]:
    """Get existing Downloader UIModule instance.
    
    Args:
        create_if_missing: Create new instance if none exists
        **kwargs: Arguments for create_downloader_uimodule if creating
        
    Returns:
        DownloaderUIModule instance or None
    """
    global _downloader_uimodule
    
    if _downloader_uimodule is None and create_if_missing:
        _downloader_uimodule = create_downloader_uimodule(**kwargs)
    
    return _downloader_uimodule

def reset_downloader_uimodule() -> None:
    """Reset global Downloader UIModule instance."""
    global _downloader_uimodule
    
    if _downloader_uimodule is not None:
        try:
            _downloader_uimodule.cleanup()
        except Exception as e:
            local_logger = get_module_logger("smartcash.ui.dataset.downloader.reset")
            local_logger.error(f"Error during cleanup: {e}")
        finally:
            _downloader_uimodule = None
    
    local_logger = get_module_logger("smartcash.ui.dataset.downloader.reset")
    local_logger.debug("🔄 Reset global Downloader UIModule instance")

def initialize_downloader_ui(config: Dict[str, Any] = None, display: bool = True):
    """Initialize and display the downloader UI.
    
    This is the main entry point for the downloader UI, typically called from a notebook cell.
    It creates the UIModule, initializes it, and displays the main widget.
    
    Args:
        config: Optional configuration dictionary
        display: Whether to display the UI immediately (True) or return components (False)
        
    Returns:
        If display=True: Returns the initialized downloader module instance
        If display=False: Returns a dictionary with module and components
    """
    try:
        # Create and initialize the downloader module
        module = create_downloader_uimodule(config=config, auto_initialize=True)
        
        if display:
            # Display mode: show UI and return module
            try:
                from IPython.display import display as ipython_display
                
                # Display the UI if it has a main widget
                if hasattr(module, 'get_main_widget'):
                    widget = module.get_main_widget()
                    if widget is not None:
                        ipython_display(widget)
                
                return module
                
            except ImportError:
                # Not in IPython environment, just return module
                return module
        else:
            # Return components mode: return dictionary with module and components
            return {
                'success': True,
                'module': module,
                'ui_components': module.get_ui_components(),
                'status': module.get_downloader_status()
            }
            
    except Exception as e:
        if display:
            # In display mode, log error and return None
            logger = get_module_logger("smartcash.ui.dataset.downloader.init")
            logger.error(f"Failed to initialize downloader UI: {e}")
            return None
        else:
            # In components mode, return error dictionary
            return {
                'success': False,
                'error': str(e),
                'module': None,
                'ui_components': {},
                'status': {}
            }

# Note: Template and shared methods are registered on-demand in create_downloader_uimodule()
# to avoid logs during import