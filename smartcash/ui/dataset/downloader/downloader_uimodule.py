"""
File: smartcash/ui/dataset/downloader/downloader_uimodule.py
Description: Dataset Downloader UIModule implementation using new core UIModule pattern.

This module refactors the dataset downloader to use the new UIModule architecture
while preserving its unique form, backend integration flow, and existing functionality.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from smartcash.ui.core.ui_module import UIModule, SharedMethodRegistry, register_operation_method
from smartcash.ui.core.ui_module_factory import create_template
from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.errors.handlers import handle_ui_errors

# Import existing downloader components and handlers
from smartcash.ui.dataset.downloader.components.downloader_ui import create_downloader_ui
from smartcash.ui.dataset.downloader.configs.downloader_config_handler import DownloaderConfigHandler
from smartcash.ui.dataset.downloader.configs.downloader_defaults import get_default_downloader_config
from smartcash.ui.dataset.downloader.operations.manager import DownloaderOperationManager
from smartcash.ui.dataset.downloader.services.downloader_service import DownloaderService

# Global module instance for singleton pattern
_downloader_uimodule: Optional[UIModule] = None

def register_downloader_template() -> None:
    """Register Downloader module template with UIModuleFactory."""
    from smartcash.ui.core.ui_module_factory import UIModuleFactory
    
    template = create_template(
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
            from smartcash.ui.dataset.downloader.services.validation_utils import validate_config
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
            return validate_config(config)
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def get_existing_dataset_count() -> Dict[str, Any]:
        """Get count of existing dataset files."""
        try:
            from smartcash.ui.dataset.downloader.services.backend_utils import get_existing_dataset_count
            logger = get_module_logger("smartcash.ui.dataset.downloader.shared")
            count = get_existing_dataset_count(logger)
            return {"success": True, "count": count}
        except Exception as e:
            return {"success": False, "error": str(e), "count": 0}
    
    def get_api_key_from_secrets() -> Dict[str, Any]:
        """Get API key from Colab secrets."""
        try:
            from smartcash.ui.dataset.downloader.services.colab_secrets import get_api_key_from_secrets
            api_key = get_api_key_from_secrets()
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
    """Dataset Downloader UIModule with enhanced functionality."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Downloader UIModule.
        
        Args:
            config: Downloader configuration (optional, uses defaults if not provided)
        """
        # Get default config and merge with provided config
        default_config = get_default_downloader_config()
        if config:
            default_config.update(config)
        
        super().__init__(
            module_name="downloader",
            parent_module="dataset", 
            config=default_config,
            auto_initialize=False
        )
        
        # Downloader-specific attributes
        self._operation_manager: Optional[DownloaderOperationManager] = None
        self._config_handler: Optional[DownloaderConfigHandler] = None
        self._downloader_service: Optional[DownloaderService] = None
        
    def initialize(self, config: Dict[str, Any] = None) -> 'DownloaderUIModule':
        """Initialize Downloader module with backend service integration.
        
        Args:
            config: Additional configuration to merge
            
        Returns:
            Self for method chaining
        """
        if config:
            self.update_config(**config)
        
        try:
            # Create UI components
            self._create_ui_components()
            
            # Setup downloader service
            self._setup_downloader_service()
            
            # Setup operation manager
            self._setup_operation_manager()
            
            # Setup config handler
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
            self._update_status("Downloader module initialized - ready for dataset operations", "success")
            
            self.logger.debug(f"✅ Initialized Downloader UIModule")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Downloader UIModule: {e}")
            self._update_status(f"Failed to initialize module: {str(e)}", "error")
            raise
        
        return self
    
    def _create_ui_components(self) -> None:
        """Create and register UI components."""
        try:
            # 1. Create UI components using existing function
            ui_components = create_downloader_ui(self.get_config())
            
            # Debug: Print detailed info about ui_components
            self.logger.info("🔍 Detailed UI components structure:")
            for key, value in ui_components.items():
                if key == 'config':
                    self.logger.info(f"  - {key}: {type(value).__name__} (keys: {list(value.keys()) if hasattr(value, 'keys') else 'N/A'})")
                else:
                    self.logger.info(f"  - {key}: {type(value).__name__}")
            
            # 2. Debug: Print all component keys before registration
            all_component_keys = list(ui_components.keys())
            self.logger.info(f"📦 UI components to register: {all_component_keys}")
            
            # 3. Extract and store action container
            action_container = ui_components.get('action_container')
            if action_container is not None:
                self._action_container = action_container
                self.logger.info("💾 Stored action container reference")
                
                # Extract buttons from action container
                buttons = {}
                if hasattr(action_container, 'get') and callable(action_container.get):
                    buttons = action_container.get('buttons', {})
                
                self.logger.info(f"🔘 Found buttons in action_container: {list(buttons.keys())}")
                
                # Add buttons directly to ui_components if not already present
                for btn_id, button in buttons.items():
                    if button is not None and btn_id not in ui_components:
                        ui_components[btn_id] = button
                        self.logger.info(f"✅ Added button to components: {btn_id}")
            
            # 4. Register all components
            registered_components = {}
            for component_type, component in ui_components.items():
                try:
                    # Skip None components
                    if component is None:
                        self.logger.warning(f"⚠️ Skipping None component: {component_type}")
                        continue
                    
                    # Register the component
                    self.register_component(component_type, component)
                    
                    # Verify registration
                    registered = self.get_component(component_type)
                    if registered is not None:
                        registered_components[component_type] = component.__class__.__name__
                        self.logger.info(f"✅ Registered: {component_type} ({component.__class__.__name__})")
                        
                        # Store direct reference for buttons
                        if component_type in ['download_button', 'check_button', 'cleanup_button']:
                            setattr(self, f'_{component_type}', component)
                            self.logger.info(f"💾 Stored direct reference to {component_type}")
                    else:
                        self.logger.error(f"❌ Failed to verify registration of: {component_type}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error registering component {component_type}: {str(e)}", exc_info=True)
            
            # 5. Verify all required buttons are registered and valid
            button_references = {}
            for btn_id in ['download_button', 'check_button', 'cleanup_button']:
                # Try to get button from multiple sources
                btn = (
                    self.get_component(btn_id) or 
                    getattr(self, f'_{btn_id}', None) or 
                    getattr(self, btn_id, None) or
                    (action_container and hasattr(action_container, 'get') and action_container.get('buttons', {}).get(btn_id))
                )
                
                if btn is not None:
                    # Ensure it's a valid button widget
                    if not hasattr(btn, 'on_click'):
                        self.logger.error(f"❌ {btn_id} is not a valid button (missing on_click)")
                        continue
                        
                    # Store the reference
                    button_references[btn_id] = btn
                    setattr(self, f'_{btn_id}', btn)
                    self.register_component(btn_id, btn)
                    
                    self.logger.info(f"✅ Verified button: {btn_id} ({type(btn).__name__})")
                else:
                    self.logger.error(f"❌ Button not found: {btn_id}")
            
            # 6. Log final status
            if len(button_references) == 3:
                self.logger.info("🎉 Successfully registered all buttons!")
            else:
                self.logger.warning(f"⚠️ Only {len(button_references)} out of 3 buttons were registered")
            
            self.logger.info(f"📊 Total registered components: {len(registered_components)}")
            
            # 7. Return the UI components for use by other methods
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create UI components: {e}", exc_info=True)
            raise
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create UI components: {e}")
            raise
    
    def _setup_downloader_service(self) -> None:
        """Setup downloader service for backend integration."""
        try:
            self._downloader_service = DownloaderService(logger=self.logger)
            self.logger.debug("🔧 Setup downloader service")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup downloader service: {e}")
            raise
    
    def _setup_operation_manager(self) -> None:
        """Setup operation manager for downloader operations."""
        try:
            # Get operation container instance directly
            operation_container = self.get_component("operation_container")
            
            # Get all UI components first
            all_components = {name: self.get_component(name) for name in self.list_components()}
            
            # Create operation manager with UI components
            self._operation_manager = DownloaderOperationManager(
                config=self.get_config(),
                operation_container=operation_container
            )
            
            # Store UI components reference in operation manager
            if hasattr(self._operation_manager, '_ui_components'):
                self._operation_manager._ui_components.update(all_components)
            else:
                self._operation_manager._ui_components = all_components
            
            # Initialize the operation manager
            self._operation_manager.initialize()
            
            # Register button callbacks if the method exists
            if hasattr(self._operation_manager, 'register_button_callbacks'):
                self._operation_manager.register_button_callbacks()
            
            self.logger.debug("⚙️ Setup operation manager with UI components")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup operation manager: {e}")
            raise
    
    def _setup_config_handler(self) -> None:
        """Setup config handler for downloader configurations."""
        try:
            self._config_handler = DownloaderConfigHandler()
            self.logger.debug("🔧 Setup config handler")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup config handler: {e}")
            raise
    
    def _register_operations(self) -> None:
        """Register downloader operations."""
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
                self.log("✅ Dataset download completed successfully!", 'info')
            else:
                error_msg = result.get("error", "Download failed")
                self._update_status(f"Download failed: {error_msg}", "error")
                self.log(f"❌ Download failed: {error_msg}", 'error')
            
            self.logger.info(f"Download completed with result: {result}")
            
        except Exception as e:
            self.logger.error(f"❌ Download button click failed: {e}")
            self._update_status(f"Download error: {str(e)}", "error")
            self.log(f"❌ Download error: {str(e)}", 'error')
    
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
                self.log(f"✅ Dataset check completed - {count} files found", 'info')
            else:
                error_msg = result.get("error", "Check failed")
                self._update_status(f"Check failed: {error_msg}", "error")
                self.log(f"❌ Check failed: {error_msg}", 'error')
            
        except Exception as e:
            self.logger.error(f"❌ Check button click failed: {e}")
            self._update_status(f"Check error: {str(e)}", "error")
            self.log(f"❌ Check error: {str(e)}", 'error')
    
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
                self.log(f"✅ Cleanup completed - {cleaned_count} items cleaned", 'info')
            else:
                error_msg = result.get("error", "Cleanup failed")
                self._update_status(f"Cleanup failed: {error_msg}", "error")
                self.log(f"❌ Cleanup failed: {error_msg}", 'error')
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup button click failed: {e}")
            self._update_status(f"Cleanup error: {str(e)}", "error")
            self.log(f"❌ Cleanup error: {str(e)}", 'error')
    
    def _extract_ui_config(self) -> Dict[str, Any]:
        """Extract configuration from UI form inputs."""
        try:
            form_widgets = self.get_component("form_widgets", {})
            
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
            self.logger.error(f"Error extracting UI config: {e}")
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
            if asyncio.iscoroutinefunction(self._operation_manager.execute_download):
                result = asyncio.run(self._operation_manager.execute_download(ui_config))
            else:
                result = self._operation_manager.execute_download(ui_config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to execute download: {e}")
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
            # Execute check operation
            import asyncio
            if asyncio.iscoroutinefunction(self._operation_manager.execute_check):
                result = asyncio.run(self._operation_manager.execute_check())
            else:
                result = self._operation_manager.execute_check()
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to execute check: {e}")
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
            # Execute cleanup operation
            import asyncio
            if asyncio.iscoroutinefunction(self._operation_manager.execute_cleanup):
                result = asyncio.run(self._operation_manager.execute_cleanup(targets))
            else:
                result = self._operation_manager.execute_cleanup(targets)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to execute cleanup: {e}")
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

# === Backward Compatibility Layer ===

@handle_ui_errors(return_type=None)
def initialize_downloader_ui(config: Dict[str, Any] = None) -> None:
    """Initialize Downloader UI using new UIModule pattern."""
    from IPython.display import display
    
    module = create_downloader_uimodule(config, auto_initialize=True)
    main_container = module.get_component('main_container')
    if main_container:
        display(main_container)

@handle_ui_errors(return_type=dict)
def get_downloader_components(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get Downloader components using new UIModule pattern."""
    module = create_downloader_uimodule(config, auto_initialize=True)
    return {
        component_type: module.get_component(component_type)
        for component_type in module.list_components()
    }

@handle_ui_errors(return_type=None)
def display_downloader_ui(config: Dict[str, Any] = None) -> None:
    """Display Downloader UI using new UIModule pattern."""
    initialize_downloader_ui(config)

# Note: Template and shared methods are registered on-demand in create_downloader_uimodule()
# to avoid logs during import