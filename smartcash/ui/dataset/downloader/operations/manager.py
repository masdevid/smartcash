"""
File: smartcash/ui/dataset/downloader/operations/manager.py
Operation manager for dataset downloader that extends OperationHandler following colab/dependency pattern
"""

from typing import Dict, Any, Optional, Callable, List
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from smartcash.ui.core.errors.handlers import handle_ui_errors
from .download_operation import DownloadOperationHandler
from .check_operation import CheckOperationHandler
from .cleanup_operation import CleanupOperationHandler

class DownloaderOperationManager(OperationHandler):
    """Operation manager for dataset downloader that extends OperationHandler."""
    
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
        """Initialize the downloader operation manager."""
        self.logger.info("🚀 Initializing Downloader operation manager")
        
        # Initialize operation handlers with UI components
        ui_components = getattr(self, '_ui_components', {})
        if hasattr(self.operation_container, 'get_ui_components'):
            ui_components.update(self.operation_container.get_ui_components())
        
        # Ensure operation container is available
        ui_components['operation_container'] = self.operation_container
        
        self.download_handler = DownloadOperationHandler(ui_components=ui_components)
        self.check_handler = CheckOperationHandler(ui_components=ui_components)
        self.cleanup_handler = CleanupOperationHandler(ui_components=ui_components)
        
        self.logger.info("✅ Downloader operation manager initialization complete")
    
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

# DownloadHandlerManager class has been moved to download_manager.py to avoid circular imports
        
        # Ensure summary container exists
        if 'summary_container' not in self.ui_components:
            from ipywidgets import HTML
            self.ui_components['summary_container'] = HTML(value="")
        
        return self.ui_components
