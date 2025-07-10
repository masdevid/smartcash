"""
File: smartcash/ui/dataset/downloader/operations/manager.py
Unified operation manager for dataset downloader with UI integration
"""

from typing import Dict, Any, Optional, Callable, List
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from smartcash.ui.core.errors.handlers import handle_ui_errors
from .download_operation import DownloadOperationHandler
from .check_operation import CheckOperationHandler
from .cleanup_operation import CleanupOperationHandler

class DownloaderOperationManager(OperationHandler):
    """Base operation manager for dataset downloader that extends OperationHandler."""
    
    def __init__(self, config: Dict[str, Any], operation_container=None, **kwargs):
        """Initialize the downloader operation manager."""
        super().__init__(
            module_name='downloader_operation_manager',
            parent_module='downloader',
            operation_container=operation_container,
            **kwargs
        )
        self.config = config
        
        # Initialize operation handlers
        self.download_handler = None
        self.check_handler = None
        self.cleanup_handler = None
    
    def initialize(self) -> None:
        """Initialize the downloader operation manager with the latest UI structure."""
        self.logger.info("🚀 Initializing Downloader operation manager")
        
        # Get UI components from instance or parent
        ui_components = getattr(self, '_ui_components', {})
        
        # Get operation container from various possible locations
        operation_container = (
            getattr(self, 'operation_container', None) or
            getattr(self, '_operation_container', None) or
            ui_components.get('operation_container')
        )
        
        # Update operation container reference if not already set
        if operation_container and 'operation_container' not in ui_components:
            ui_components['operation_container'] = operation_container
            
        # Store the updated components back to the instance
        self._ui_components = ui_components
        
        # Initialize operation handlers with the updated components
        self.download_handler = DownloadOperationHandler(ui_components=ui_components)
        self.check_handler = CheckOperationHandler(ui_components=ui_components)
        self.cleanup_handler = CleanupOperationHandler(ui_components=ui_components)
        
        # Store references to buttons if they exist in the UI components
        self._buttons = {
            'download': ui_components.get('download_button'),
            'check': ui_components.get('check_button'),
            'cleanup': ui_components.get('cleanup_button')
        }
        
        self.logger.info("✅ Downloader operation manager initialization complete")
    
    def handle_error(self, error_msg: str, **kwargs):
        """Handle errors with proper logging and UI feedback."""
        self.logger.error(f"❌ {error_msg}")
        if hasattr(self, 'operation_container') or hasattr(self, '_operation_container'):
            # Log error to operation container if available
            operation_container = getattr(self, 'operation_container', None) or getattr(self, '_operation_container', None)
            if operation_container and hasattr(operation_container, 'log_error'):
                operation_container.log_error(error_msg)
    
    def log_error(self, error_msg: str):
        """Log error to operation container."""
        self.handle_error(error_msg)
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations."""
        return {
            'download': self.execute_download,
            'check': self.execute_check,
            'cleanup': self.execute_cleanup
        }
    
    async def execute_download(self, ui_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dataset download operation."""
        try:
            if not self.download_handler:
                self.initialize()
            
            # Execute the actual download
            result = self.download_handler.execute_download(ui_config)
            
            # Update operation summary with results
            await self._update_operation_summary('download', result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in execute_download: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_check(self) -> Dict[str, Any]:
        """Execute dataset check operation."""
        try:
            if not self.check_handler:
                self.initialize()
            
            # Execute the actual check
            result = self.check_handler.execute_check()
            
            # Update operation summary with results
            await self._update_operation_summary('check', result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in execute_check: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_cleanup(self, targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute dataset cleanup operation."""
        try:
            if not self.cleanup_handler:
                self.initialize()
            
            # Get cleanup targets if not provided
            if targets is None:
                targets_result = self.cleanup_handler.get_cleanup_targets()
                if not targets_result.get('success'):
                    return targets_result
                targets = targets_result.get('targets', [])
            
            # Execute the actual cleanup
            result = self.cleanup_handler.execute_cleanup({'targets': targets})
            
            # Update operation summary with results
            await self._update_operation_summary('cleanup', result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in execute_cleanup: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _update_operation_summary(self, operation_type: str, result: Dict[str, Any]) -> None:
        """Update operation summary with operation results."""
        try:
            from ..components.operation_summary import update_operation_summary
            
            # Get the operation summary widget from UI components
            ui_components = getattr(self, '_ui_components', {})
            if hasattr(self.operation_container, 'get_ui_components'):
                ui_components.update(self.operation_container.get_ui_components())
            
            operation_summary = ui_components.get('operation_summary')
            if operation_summary:
                # Determine status type based on result
                if result.get('cancelled'):
                    status_type = 'warning'
                elif result.get('success'):
                    status_type = 'success'
                else:
                    status_type = 'error'
                
                # Update the summary widget
                update_operation_summary(operation_summary, operation_type, result, status_type)
                self.logger.info(f"✅ Updated operation summary for {operation_type}")
            else:
                self.logger.warning("⚠️ Operation summary widget not found in UI components")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update operation summary: {e}")


class DownloadHandlerManager(DownloaderOperationManager):
    """Enhanced operation manager with UI integration and button management."""
    
    @handle_ui_errors(error_component_title="Download Handler Manager Error", log_error=True)
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize download handler manager with UI components.
        
        Args:
            ui_components: Dictionary UI components
            config: Configuration dictionary
            **kwargs: Additional arguments passed to parent class
        """
        # Initialize parent with proper parameters
        config = config or {}
        operation_container = ui_components.get('operation_container') if ui_components else None
        super().__init__(config=config, operation_container=operation_container, **kwargs)
        
        # Store UI components
        self._ui_components = ui_components or {}
        
        # Initialize operation handlers
        self.initialize()
        
        # Button references for enabling/disabling during operations
        self._buttons = {}
        self._get_button_references()
    
    def _get_button_references(self) -> None:
        """Get references to buttons for enabling/disabling during operations."""
        if not self._ui_components:
            self.logger.warning("⚠️ No UI components available for button references")
            return
            
        # Get button references from UI components
        button_keys = ['download_button', 'check_button', 'cleanup_button']
        for key in button_keys:
            if key in self._ui_components:
                self._buttons[key] = self._ui_components[key]
                
        self.logger.debug(f"📋 Found {len(self._buttons)} button references")
    
    def disable_buttons(self) -> None:
        """Disable buttons during operation execution."""
        for button in self._buttons.values():
            if hasattr(button, 'disabled'):
                button.disabled = True
    
    def enable_buttons(self) -> None:
        """Enable buttons after operation execution."""
        for button in self._buttons.values():
            if hasattr(button, 'disabled'):
                button.disabled = False
    
    def _reset_progress_tracker(self) -> None:
        """Reset progress tracker after operation."""
        operation_container = getattr(self, 'operation_container', None) or getattr(self, '_operation_container', None)
        if operation_container and hasattr(operation_container, 'reset_progress'):
            operation_container.reset_progress()
    
    @handle_ui_errors(error_component_title="Download Operation Error", log_error=True)
    def _execute_download_operation(self) -> None:
        """Execute download operation with UI feedback."""
        # Disable buttons during operation
        self.disable_buttons()
        
        try:
            # Execute download
            self.download_handler.execute_download()
            
            # Reset progress tracker
            self._reset_progress_tracker()
            
        finally:
            # Always enable buttons after operation
            self.enable_buttons()
    
    @handle_ui_errors(error_component_title="Check Operation Error", log_error=True)
    def _execute_check_operation(self) -> None:
        """Execute check operation with UI feedback."""
        # Disable buttons during operation
        self.disable_buttons()
        
        try:
            # Execute check
            self.check_handler.execute_check()
            
            # Reset progress tracker
            self._reset_progress_tracker()
            
        finally:
            # Always enable buttons after operation
            self.enable_buttons()
    
    @handle_ui_errors(error_component_title="Cleanup Operation Error", log_error=True)
    def _execute_cleanup_operation(self, targets: List[str]) -> None:
        """Execute cleanup operation with UI feedback.
        
        Args:
            targets: List of targets to clean up
        """
        # Disable buttons during operation
        self.disable_buttons()
        
        try:
            # Execute cleanup
            self.cleanup_handler.execute_cleanup(targets)
            
            # Reset progress tracker
            self._reset_progress_tracker()
            
        finally:
            # Always enable buttons after operation
            self.enable_buttons()
    
    def register_button_callbacks(self) -> None:
        """Register button callbacks for download, check, and cleanup operations."""
        try:
            # Register download button callback
            if 'download_button' in self._ui_components:
                self._ui_components['download_button'].on_click(
                    lambda _: self._execute_download_operation()
                )
                self.logger.info("✅ Registered download button callback")
            
            # Register check button callback
            if 'check_button' in self._ui_components:
                self._ui_components['check_button'].on_click(
                    lambda _: self._execute_check_operation()
                )
                self.logger.info("✅ Registered check button callback")
            
            # Register cleanup button callback with targets
            if 'cleanup_button' in self._ui_components and 'cleanup_targets' in self._ui_components:
                self._ui_components['cleanup_button'].on_click(
                    lambda _: self._execute_cleanup_operation(
                        self._ui_components['cleanup_targets'].value
                    )
                )
                self.logger.info("✅ Registered cleanup button callback")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to register button callbacks: {e}")
