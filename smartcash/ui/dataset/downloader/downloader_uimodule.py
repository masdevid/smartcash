"""
Dataset Downloader UIModule - New Core Pattern
Following new UIModule architecture with clean implementation.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime

from smartcash.ui.core.ui_module import UIModule
from smartcash.ui.core.decorators import suppress_all_init_logs
from smartcash.ui.dataset.downloader.components.downloader_ui import create_downloader_ui
from smartcash.ui.dataset.downloader.configs.downloader_defaults import get_default_downloader_config
from smartcash.ui.dataset.downloader.operations.manager import DownloaderOperationManager
from smartcash.ui.dataset.downloader.services.downloader_service import DownloaderService

class DownloaderUIModule(UIModule):
    """Dataset Downloader UIModule following new core pattern."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize downloader module."""
        super().__init__(
            module_name='downloader',
            parent_module='dataset',
            config=config or get_default_downloader_config(),
            auto_initialize=False
        )
        
        self._operation_manager: Optional[DownloaderOperationManager] = None
        self._downloader_service: Optional[DownloaderService] = None
    
    @suppress_all_init_logs(duration=5.0)
    def initialize(self, config: Dict[str, Any] = None) -> 'DownloaderUIModule':
        """Initialize downloader module."""
        try:
            if config:
                self.update_config(**config)
            
            # Create UI components
            ui_components = create_downloader_ui(self.get_config())
            
            # Register components
            for name, component in ui_components.items():
                if component is not None:
                    self.register_component(name, component)
            
            # Initialize operation manager
            operation_container = self.get_component('operation_container')
            if operation_container:
                self._operation_manager = DownloaderOperationManager(
                    config=self.get_config(),
                    operation_container=operation_container
                )
                self._operation_manager.initialize()
            
            # Initialize downloader service
            self._downloader_service = DownloaderService(self.logger)
            
            # Register operations
            self._register_operations()
            
            # Setup event handlers
            self._setup_event_handlers()
            
            # Initialize parent
            super().initialize()
            
            # Log initialization completion
            self._log_initialization_complete()
            
            return self
            
        except Exception as e:
            self.logger.error(f"Failed to initialize downloader module: {e}")
            raise RuntimeError(f"Downloader initialization failed: {e}") from e
    
    def _log_initialization_complete(self) -> None:
        """Log initialization completion to operation container."""
        if self._operation_manager and hasattr(self._operation_manager, 'log'):
            self._operation_manager.log("✅ Downloader module initialized successfully", 'info')
            self._operation_manager.log("📥 Ready for dataset download operations", 'info')
            
            # Log available operations
            if hasattr(self._operation_manager, 'get_operations'):
                operations = self._operation_manager.get_operations()
                self._operation_manager.log(f"📋 Available operations: {', '.join(operations.keys())}", 'info')
    
    def _register_operations(self) -> None:
        """Register operations with the module."""
        if not self._operation_manager:
            return
            
        operations = self._operation_manager.get_operations()
        for op_name, op_func in operations.items():
            self.register_operation(op_name, op_func)
        
        # Register additional operations
        self.register_operation("validate_config", self.validate_configuration)
        self.register_operation("get_dataset_count", self.get_existing_dataset_count)
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for buttons."""
        button_handlers = {
            'download_button': self._handle_download_click,
            'check_button': self._handle_check_click,
            'cleanup_button': self._handle_cleanup_click
        }
        
        for button_id, handler in button_handlers.items():
            button = self.get_component(button_id)
            if button and hasattr(button, 'on_click'):
                button.on_click(handler)
    
    def _handle_download_button_click(self, button=None) -> None:
        """Handle download button click (alias for compatibility)."""
        return self._handle_download_click(button)
    
    def _handle_check_button_click(self, button=None) -> None:
        """Handle check button click (alias for compatibility)."""
        return self._handle_check_click(button)
    
    def _handle_cleanup_button_click(self, button=None) -> None:
        """Handle cleanup button click (alias for compatibility)."""
        return self._handle_cleanup_click(button)
    
    def _handle_download_click(self, button=None) -> None:
        """Handle download button click."""
        try:
            # Disable all operation buttons
            self._set_buttons_enabled(False)
            
            self._update_status("Starting dataset download...", "info")
            self.log("📥 Starting dataset download operation", "info")
            
            config = self._extract_ui_config()
            result = self.execute_download(config)
            
            if result.get("success", False):
                self._update_status("Dataset download completed successfully!", "success")
                self.log("✅ Dataset download completed successfully", "success")
            else:
                error_msg = result.get("error", "Download failed")
                self._update_status(f"Download failed: {error_msg}", "error")
                self.log(f"❌ Download failed: {error_msg}", "error")
                
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            self._update_status(f"Download error: {e}", "error")
            self.log(f"❌ Download error: {e}", "error")
        finally:
            # Re-enable buttons
            self._set_buttons_enabled(True)
    
    def _handle_check_click(self, button=None) -> None:
        """Handle check button click."""
        try:
            # Disable all operation buttons
            self._set_buttons_enabled(False)
            
            self._update_status("Checking dataset status...", "info")
            self.log("🔍 Starting dataset check operation", "info")
            
            # Create a task to run the async operation
            async def run_check():
                try:
                    result = await self.execute_check()
                    
                    if result.get("success", False):
                        count = result.get("count", 0)
                        self._update_status(f"Dataset check completed - {count} files found", "success")
                        self.log(f"✅ Dataset check completed - {count} files found", "success")
                    else:
                        error_msg = result.get("error", "Check failed")
                        self._update_status(f"Check failed: {error_msg}", "error")
                        self.log(f"❌ Check failed: {error_msg}", "error")
                except Exception as e:
                    self.logger.error(f"Error in check operation: {e}", exc_info=True)
                    self._update_status(f"Check error: {str(e)}", "error")
                    self.log(f"❌ Check error: {str(e)}", "error")
                finally:
                    # Re-enable buttons when done
                    self._set_buttons_enabled(True)
            
            # Start the async task
            import asyncio
            asyncio.create_task(run_check())
            
        except Exception as e:
            self.logger.error(f"Failed to start check operation: {e}", exc_info=True)
            self._update_status(f"Failed to start check: {str(e)}", "error")
            self._set_buttons_enabled(True)
                
        except Exception as e:
            self.logger.error(f"Check failed: {e}")
            self._update_status(f"Check error: {e}", "error")
            self.log(f"❌ Check error: {e}", "error")
        finally:
            # Re-enable buttons
            self._set_buttons_enabled(True)
    
    def _handle_cleanup_click(self, button=None) -> None:
        """Handle cleanup button click."""
        try:
            # Disable all operation buttons
            self._set_buttons_enabled(False)
            
            self._update_status("Starting dataset cleanup...", "info")
            self.log("🧹 Starting dataset cleanup operation", "info")
            
            result = self.execute_cleanup()
            
            if result.get("success", False):
                cleaned_count = result.get("cleaned_count", 0)
                self._update_status(f"Cleanup completed - {cleaned_count} items cleaned", "success")
                self.log(f"✅ Cleanup completed - {cleaned_count} items cleaned", "success")
            else:
                error_msg = result.get("error", "Cleanup failed")
                self._update_status(f"Cleanup failed: {error_msg}", "error")
                self.log(f"❌ Cleanup failed: {error_msg}", "error")
                
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            self._update_status(f"Cleanup error: {e}", "error")
            self.log(f"❌ Cleanup error: {e}", "error")
        finally:
            # Re-enable buttons
            self._set_buttons_enabled(True)
    
    def _extract_ui_config(self) -> Dict[str, Any]:
        """Extract configuration from UI form inputs."""
        # Get form widgets directly from registered components
        workspace_input = self.get_component('workspace_input')
        project_input = self.get_component('project_input')
        version_input = self.get_component('version_input')
        api_key_input = self.get_component('api_key_input')
        validate_checkbox = self.get_component('validate_checkbox')
        backup_checkbox = self.get_component('backup_checkbox')
        
        return {
            'workspace': getattr(workspace_input, 'value', ''),
            'project': getattr(project_input, 'value', ''),
            'version': getattr(version_input, 'value', ''),
            'api_key': getattr(api_key_input, 'value', ''),
            'validate_download': getattr(validate_checkbox, 'value', True),
            'backup_existing': getattr(backup_checkbox, 'value', False),
        }
    
    def _update_status(self, message: str, status_type: str = "info") -> None:
        """Update status panel in header container."""
        header_container = self.get_component("header_container")
        if header_container and hasattr(header_container, 'update_status'):
            header_container.update_status(message, status_type)
    
    def _set_buttons_enabled(self, enabled: bool) -> None:
        """Enable or disable operation buttons."""
        button_ids = ['download_button', 'check_button', 'cleanup_button']
        
        for button_id in button_ids:
            button = self.get_component(button_id)
            if button and hasattr(button, 'disabled'):
                button.disabled = not enabled
                
        # Also update button appearance based on state
        if not enabled:
            self.log("🔒 Operation buttons disabled during processing", "info")
        else:
            self.log("🔓 Operation buttons re-enabled", "info")
    
    def execute_download(self, ui_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute dataset download."""
        import asyncio
        from IPython import get_ipython
        
        if not self._operation_manager:
            return {"success": False, "error": "Operation manager not initialized"}
        
        try:
            config = ui_config or self._extract_ui_config()
            
            # Check if running in Jupyter for async support
            ipython = get_ipython()
            in_jupyter = ipython is not None
            
            if in_jupyter and asyncio.iscoroutinefunction(self._operation_manager.execute_download):
                # Handle async operations in Jupyter
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create new task in existing loop
                        import nest_asyncio
                        nest_asyncio.apply()
                        result = loop.run_until_complete(self._operation_manager.execute_download(config))
                    else:
                        result = asyncio.run(self._operation_manager.execute_download(config))
                except Exception as async_error:
                    result = {"success": False, "error": f"Async execution failed: {async_error}"}
            else:
                # Synchronous execution
                result = self._operation_manager.execute_download(config)
                
            return result
        except Exception as e:
            self.logger.error(f"Download execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_check(self) -> Dict[str, Any]:
        """Execute dataset check.
        
        Returns:
            Dictionary containing check results
        """
        if not self._operation_manager:
            return {"success": False, "error": "Operation manager not initialized"}
        
        try:
            return await self._operation_manager.execute_check()
        except Exception as e:
            self.logger.error(f"Check execution failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def execute_cleanup(self, targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute dataset cleanup."""
        if not self._operation_manager:
            return {"success": False, "error": "Operation manager not initialized"}
        
        try:
            return self._operation_manager.execute_cleanup(targets)
        except Exception as e:
            self.logger.error(f"Cleanup execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_configuration(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate downloader configuration."""
        if not self._downloader_service:
            return {"valid": False, "error": "Downloader service not initialized"}
        
        try:
            config = config or self.get_config()
            return self._downloader_service.validate_config(config)
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return {"valid": False, "error": str(e)}
    
    def get_existing_dataset_count(self) -> Dict[str, Any]:
        """Get count of existing dataset files."""
        if not self._downloader_service:
            return {"success": False, "count": 0, "error": "Downloader service not initialized"}
        
        try:
            count = self._downloader_service.get_existing_dataset_count()
            return {"success": True, "count": count}
        except Exception as e:
            self.logger.error(f"Failed to get dataset count: {e}")
            return {"success": False, "count": 0, "error": str(e)}
    
    def reset_downloader(self) -> None:
        """Reset downloader to default state."""
        try:
            # Reset UI to default values
            config = get_default_downloader_config()
            self.update_config(**config)
            
            # Update status
            self._update_status("Downloader reset to defaults", "info")
        except Exception as e:
            self.logger.error(f"Failed to reset downloader: {e}")
            self._update_status(f"Reset failed: {e}", "error")
    
    def get_config_handler(self):
        """Get config handler instance."""
        if not hasattr(self, '_config_handler'):
            from smartcash.ui.dataset.downloader.configs.downloader_config_handler import DownloaderConfigHandler
            self._config_handler = DownloaderConfigHandler()
        return self._config_handler
    
    def log(self, message: str, level: str = 'info') -> None:
        """Log message to operation container."""
        if self._operation_manager and hasattr(self._operation_manager, 'log'):
            self._operation_manager.log(message, level)
        else:
            getattr(self.logger, level, self.logger.info)(message)
    
    def get_operation_manager(self):
        """Get operation manager instance."""
        return self._operation_manager
    
    def get_ui_components(self) -> Dict[str, Any]:
        """Get all UI components."""
        return {name: self.get_component(name) for name in self.list_components()}
    
    def create_ui_components(self) -> Dict[str, Any]:
        """Create and return UI components."""
        return self.get_ui_components()
    
    def get_downloader_service(self):
        """Get downloader service instance."""
        return self._downloader_service
    
    def _setup_downloader_service(self) -> None:
        """Setup downloader service."""
        if not self._downloader_service:
            self._downloader_service = DownloaderService(self.logger)
    
    def setup_downloader_service(self) -> None:
        """Setup downloader service (public alias)."""
        return self._setup_downloader_service()
    
    def track_status(self, status: str, details: str = "") -> None:
        """Track operation status."""
        self._update_status(f"{status}: {details}" if details else status, "info")
    
    async def async_execute_download(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute download operation asynchronously."""
        if not self._operation_manager:
            return {"success": False, "error": "Operation manager not initialized"}
        
        try:
            config = config or self._extract_ui_config()
            return await self._operation_manager.execute_download(config)
        except Exception as e:
            self.logger.error(f"Async download execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_jupyter_environment(self) -> Dict[str, Any]:
        """Handle Jupyter-specific environment setup."""
        try:
            # Check if running in Jupyter
            in_jupyter = False
            try:
                from IPython import get_ipython
                ipython = get_ipython()
                in_jupyter = ipython is not None and ipython.__class__.__name__ == 'ZMQInteractiveShell'
            except ImportError:
                pass
            
            return {
                "success": True,
                "in_jupyter": in_jupyter,
                "environment": "jupyter" if in_jupyter else "standard"
            }
        except Exception as e:
            self.logger.error(f"Failed to handle Jupyter environment: {e}")
            return {"success": False, "error": str(e), "in_jupyter": False}
    
    def get_downloader_status(self) -> Dict[str, Any]:
        """Get current downloader status."""
        try:
            return {
                "success": True,
                "status": "ready",
                "operation_manager_initialized": self._operation_manager is not None,
                "downloader_service_initialized": self._downloader_service is not None,
                "module_initialized": self.is_initialized(),
                "components_count": len(self.list_components())
            }
        except Exception as e:
            self.logger.error(f"Failed to get downloader status: {e}")
            return {"success": False, "error": str(e)}
    
    def get_main_widget(self):
        """Get the main widget for display."""
        main_container = self.get_component('main_container')
        if hasattr(main_container, 'container'):
            return main_container.container
        return main_container

# Global module instance for singleton pattern
_downloader_module_instance: Optional[DownloaderUIModule] = None

def create_downloader_uimodule(config: Dict[str, Any] = None, auto_initialize: bool = False) -> DownloaderUIModule:
    """Create new downloader UIModule instance."""
    module = DownloaderUIModule(config)
    if auto_initialize:
        module.initialize()
    return module

def get_downloader_uimodule() -> Optional[DownloaderUIModule]:
    """Get current downloader UIModule instance."""
    global _downloader_module_instance
    return _downloader_module_instance

def reset_downloader_uimodule() -> None:
    """Reset downloader UIModule instance."""
    global _downloader_module_instance
    _downloader_module_instance = None

# Factory and convenience functions
def initialize_downloader_ui(config: Dict[str, Any] = None, display: bool = True):
    """Initialize and display the downloader UI."""
    try:
        # Create and initialize module
        module = DownloaderUIModule(config)
        module.initialize()
        
        if display:
            # Display mode: show UI and return module
            try:
                from IPython.display import display as ipython_display
                widget = module.get_main_widget()
                if widget is not None:
                    ipython_display(widget)
                return module
            except ImportError:
                return module
        else:
            # Return components mode
            return {
                'success': True,
                'module': module,
                'ui_components': {name: module.get_component(name) for name in module.list_components()},
                'status': {'initialized': True, 'module_name': module.module_name}
            }
            
    except Exception as e:
        if display:
            return None
        else:
            return {
                'success': False,
                'error': str(e),
                'module': None,
                'ui_components': {},
                'status': {}
            }