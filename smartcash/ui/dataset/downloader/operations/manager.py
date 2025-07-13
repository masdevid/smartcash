"""
File: smartcash/ui/dataset/downloader/operations/manager.py
Unified operation manager for dataset downloader with UI integration
"""

import asyncio
from typing import Dict, Any, Optional, Callable, List
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from smartcash.ui.core.errors.handlers import handle_ui_errors
from smartcash.ui.logger import get_module_logger
from .download_operation import DownloadOperationHandler
from .check_operation import CheckOperationHandler
from .cleanup_operation import CleanupOperationHandler

class DownloaderOperationManager(OperationHandler):
    """
    Operation manager for dataset downloader that extends OperationHandler.
    
    Features:
    - 📥 Dataset download operations
    - 🔍 Dataset verification and validation
    - 🧹 Cleanup operations
    - 📊 Progress tracking and logging integration
    - 🛡️ Error handling with user feedback
    - 🎯 Button management with disable/enable functionality
    """
    
    def __init__(self, config: Dict[str, Any], operation_container=None, **kwargs):
        """
        Initialize the downloader operation manager.
        
        Args:
            config: Configuration dictionary for the downloader
            operation_container: UI operation container for logging and progress
            **kwargs: Additional arguments passed to parent class
        """
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
        
        # Store UI components
        self._ui_components = {}
        
        # Initialize logger
        self.logger = get_module_logger("smartcash.ui.dataset.downloader.operations")
    
    def initialize(self) -> None:
        """Initialize the downloader operation manager with the latest UI structure."""
        try:
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
            
            # Initialize parent class (this will set up logging and other base functionality)
            super().initialize()
            
            # Log to operation container
            self.log("🚀 Initializing Downloader operation manager", 'info')
            
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
            
            self.log("✅ Downloader operation manager initialization complete", 'info')
            
        except Exception as e:
            error_msg = f"❌ Failed to initialize downloader operation manager: {str(e)}"
            self.logger.error(error_msg)
            self.log(error_msg, 'error')
            raise
    
    def log(self, message: str, level: str = 'info', **kwargs) -> None:
        """
        Log a message with the specified level.
        
        Args:
            message: The message to log
            level: Log level ('debug', 'info', 'warning', 'error', 'critical')
            **kwargs: Additional arguments to pass to the logger
        """
        # Log to the operation container if available
        if hasattr(self, '_operation_container') and self._operation_container:
            self._operation_container.log(message, level=level)
        
        # Also log to the standard logger
        log_method = getattr(self.logger, level, self.logger.info)
        log_method(message, **kwargs)
    
    def handle_error(self, error_msg: str, **kwargs) -> None:
        """
        Handle errors with proper logging and UI feedback.
        
        Args:
            error_msg: The error message to log
            **kwargs: Additional arguments to pass to the logger
        """
        self.log(f"❌ {error_msg}", 'error', **kwargs)
    
    def log_error(self, error_msg: str, **kwargs) -> None:
        """
        Log an error message.
        
        Args:
            error_msg: The error message to log
            **kwargs: Additional arguments to pass to the logger
        """
        self.handle_error(error_msg, **kwargs)
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations."""
        return {
            'download': self.execute_download,
            'check': self.execute_check,
            'cleanup': self.execute_cleanup
        }
    
    async def execute_download(self, ui_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute dataset download operation with proper logging and error handling.
        
        Args:
            ui_config: Configuration dictionary from the UI
            
        Returns:
            Dictionary containing download results
            
        Raises:
            RuntimeError: If the download operation fails
        """
        self.log("Starting dataset download operation...", 'info')
        
        try:
            if not self.download_handler:
                self.log("Initializing download handler...", 'debug')
                self.initialize()
            
            # Log the start of the download operation
            self.log(f"Downloading dataset with config: {ui_config}", 'info')
            
            # Execute the actual download
            result = await asyncio.to_thread(
                self.download_handler.execute_download,
                ui_config
            )
            
            # Update operation summary with results
            await self._update_operation_summary('download', result)
            
            # Log success
            self.log("Dataset download completed successfully", 'success')
            
            return result
            
        except Exception as e:
            error_msg = f"Error during download operation: {str(e)}"
            self.log(error_msg, 'error')
            self.logger.exception("Download operation failed")
            raise RuntimeError(error_msg) from e
    
    async def execute_check(self) -> Dict[str, Any]:
        """
        Execute dataset check operation with proper logging and error handling.
        
        Returns:
            Dictionary containing check results
            
        Raises:
            RuntimeError: If the check operation fails
        """
        self.log("🔍 Starting dataset check operation...", 'info')
        
        try:
            if not self.check_handler:
                self.log("Initializing check handler...", 'debug')
                self.initialize()
            
            # Execute the actual check
            self.log("Verifying dataset integrity...", 'info')
            result = await asyncio.to_thread(self.check_handler.execute_check)
            
            # Update operation summary with results
            await self._update_operation_summary('check', result)
            
            # Log success
            if result.get('success', False):
                self.log("✅ Dataset check completed successfully", 'success')
            else:
                self.log("⚠️ Dataset check completed with issues", 'warning')
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error during check operation: {str(e)}"
            self.log(error_msg, 'error')
            self.logger.exception("Check operation failed")
            raise RuntimeError(error_msg) from e
    
    async def execute_cleanup(self, targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute dataset cleanup operation."""
        try:
            if not self.cleanup_handler:
                self.log("Initializing cleanup handler...", 'debug')
                self.initialize()
            
            # Execute the actual cleanup
            self.log("Cleaning up dataset files...", 'info')
            result = await asyncio.to_thread(
                self.cleanup_handler.execute_cleanup,
                targets
            )
            
            # Update operation summary with results
            await self._update_operation_summary('cleanup', result)
            
            # Log success
            if result.get('success', False):
                self.log("✅ Cleanup completed successfully", 'success')
                
                # Log details about what was cleaned up
                if 'deleted' in result and result['deleted']:
                    self.log(f"🗑️ Deleted {len(result['deleted'])} files/directories", 'info')
                if 'errors' in result and result['errors']:
                    self.log(f"⚠️ Encountered {len(result['errors'])} errors during cleanup", 'warning')
            else:
                self.log("⚠️ Cleanup completed with issues", 'warning')
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error during cleanup operation: {str(e)}"
            self.log(error_msg, 'error')
            self.logger.exception("Cleanup operation failed")
            raise RuntimeError(error_msg) from e
    
    async def _update_operation_summary(self, operation_type: str, result: Dict[str, Any]) -> None:
        """
        Update operation summary with operation results.
        
        Args:
            operation_type: Type of operation ('download', 'check', 'cleanup')
            result: Dictionary containing operation results
        """
        try:
            if not hasattr(self, '_operation_container') or not self._operation_container:
                self.log("No operation container available for summary update", 'debug')
                return
            
            # Create a more detailed summary
            summary = f"## {operation_type.capitalize()} Operation "
            
            # Add status with appropriate emoji
            if result.get('success', False):
                summary += "✅ Completed Successfully\n\n"
                
                # Add success message if available
                if 'message' in result and result['message']:
                    summary += f"{result['message']}\n\n"
            else:
                summary += "❌ Failed\n\n"
                
                # Add error message if available
                if 'error' in result and result['error']:
                    summary += f"**Error:** {result['error']}\n\n"
                elif 'message' in result and result['message']:
                    summary += f"**Message:** {result['message']}\n\n"
            
            # Add additional details if available
            if 'details' in result and result['details']:
                summary += "### Details\n"
                if isinstance(result['details'], dict):
                    for key, value in result['details'].items():
                        summary += f"- **{key.replace('_', ' ').title()}:** {value}\n"
                else:
                    summary += f"{result['details']}\n"
            
            # Add statistics if available
            if 'stats' in result and isinstance(result['stats'], dict):
                summary += "\n### Statistics\n"
                for key, value in result['stats'].items():
                    summary += f"- **{key.replace('_', ' ').title()}:** {value}\n"
            
            # Log the summary update
            self.log(f"Updating {operation_type} operation summary", 'debug')
            
            # Update the operation container
            if hasattr(self._operation_container, 'update_summary'):
                self._operation_container.update_summary(summary)
            else:
                self.log("Operation container does not have update_summary method", 'warning')
            
        except Exception as e:
            error_msg = f"Failed to update operation summary: {str(e)}"
            self.log(error_msg, 'error')
            self.logger.exception("Error in _update_operation_summary")


class DownloadHandlerManager(DownloaderOperationManager):
    """
    Enhanced operation manager with UI integration and button management.
    
    This class extends DownloaderOperationManager with additional UI integration
    features like button state management and enhanced error handling.
    """
    
    @handle_ui_errors(error_component_title="Download Handler Manager Error", log_error=True)
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, 
                 config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize download handler manager with UI components.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration dictionary
            **kwargs: Additional arguments passed to parent class
        """
        # Initialize parent with proper parameters
        config = config or {}
        operation_container = ui_components.get('operation_container') if ui_components else None
        
        # Store UI components before parent init so they're available during initialization
        self._ui_components = ui_components or {}
        
        # Initialize parent class
        super().__init__(config=config, operation_container=operation_container, **kwargs)
        
        # Initialize operation handlers
        try:
            self.initialize()
            
            # Set up button references and callbacks
            self._buttons = {}
            self._get_button_references()
            
            # Register button callbacks
            self.register_button_callbacks()
            
            self.log("✅ Download handler manager initialized successfully", 'info')
            
        except Exception as e:
            error_msg = f"❌ Failed to initialize download handler manager: {str(e)}"
            self.log(error_msg, 'error')
            raise RuntimeError(error_msg) from e
    
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
    def _execute_cleanup_operation(self, targets: List[str]) -> Dict[str, Any]:
        """Execute cleanup operation with UI feedback.
        
        Args:
            targets: List of targets to clean up
            
        Returns:
            Dictionary with cleanup results
        """
        # Disable buttons during operation
        self.disable_buttons()
        
        try:
            # Execute cleanup and get result
            result = self.cleanup_handler.execute_cleanup(targets)
            
            # Reset progress tracker
            self._reset_progress_tracker()
            
            return result
            
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
