"""
Downloader Operation Manager - Clean New Core Pattern
Simplified operation manager without backward compatibility.
"""
import asyncio
from typing import Dict, Any, Optional, List

from smartcash.ui.core.handlers.operation_handler import OperationHandler
from .download_operation import DownloadOperationHandler
from .check_operation import CheckOperationHandler
from .cleanup_operation import CleanupOperationHandler

class DownloaderOperationManager(OperationHandler):
    """Operation manager for dataset downloader."""
    
    def __init__(self, config: Dict[str, Any], operation_container=None):
        """Initialize operation manager."""
        super().__init__(
            module_name='downloader',
            parent_module='dataset',
            operation_container=operation_container
        )
        self.config = config
        self.download_handler = None
        self.check_handler = None
        self.cleanup_handler = None
    
    def initialize(self) -> None:
        """Initialize operation manager."""
        try:
            super().initialize()
            
            # Initialize operation handlers
            ui_components = {'operation_container': self._operation_container}
            self.download_handler = DownloadOperationHandler(ui_components=ui_components)
            self.check_handler = CheckOperationHandler(ui_components=ui_components)
            self.cleanup_handler = CleanupOperationHandler(ui_components=ui_components)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize operation manager: {e}")
            raise
    
    def get_operations(self) -> Dict[str, str]:
        """Get available operations."""
        return {
            'download': 'Download dataset from Roboflow',
            'check': 'Check existing dataset status',
            'cleanup': 'Clean up dataset files'
        }
    
    async def execute_download(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute download operation."""
        if not self.download_handler:
            return {'success': False, 'error': 'Download handler not initialized'}
        
        try:
            self.log("📥 Starting dataset download", 'info')
            self.update_progress(0, "Initializing download...")
            
            # Execute download with progress tracking
            result = await self.download_handler.execute(config)
            
            if result.get('success'):
                self.update_progress(100, "Download completed")
                self.log("✅ Download completed successfully", 'success')
            else:
                self.update_progress(0, "Download failed")
                error = result.get('error', 'Unknown error')
                self.log(f"❌ Download failed: {error}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Download operation failed: {e}")
            self.log(f"❌ Download error: {e}", 'error')
            return {'success': False, 'error': str(e)}
    
    async def execute_check(self) -> Dict[str, Any]:
        """Execute check operation."""
        if not self.check_handler:
            return {'success': False, 'error': 'Check handler not initialized'}
        
        try:
            self.log("🔍 Checking dataset status", 'info')
            self.update_progress(0, "Checking dataset...")
            
            result = await self.check_handler.execute()
            
            if result.get('success'):
                count = result.get('count', 0)
                self.update_progress(100, f"Check completed - {count} files")
                self.log(f"✅ Check completed - {count} files found", 'success')
            else:
                self.update_progress(0, "Check failed")
                error = result.get('error', 'Unknown error')
                self.log(f"❌ Check failed: {error}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Check operation failed: {e}")
            self.log(f"❌ Check error: {e}", 'error')
            return {'success': False, 'error': str(e)}
    
    async def execute_cleanup(self, targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute cleanup operation."""
        if not self.cleanup_handler:
            return {'success': False, 'error': 'Cleanup handler not initialized'}
        
        try:
            self.log("🧹 Starting dataset cleanup", 'info')
            self.update_progress(0, "Cleaning up...")
            
            result = await self.cleanup_handler.execute(targets)
            
            if result.get('success'):
                cleaned = result.get('cleaned_count', 0)
                self.update_progress(100, f"Cleanup completed - {cleaned} items")
                self.log(f"✅ Cleanup completed - {cleaned} items cleaned", 'success')
            else:
                self.update_progress(0, "Cleanup failed")
                error = result.get('error', 'Unknown error')
                self.log(f"❌ Cleanup failed: {error}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Cleanup operation failed: {e}")
            self.log(f"❌ Cleanup error: {e}", 'error')
            return {'success': False, 'error': str(e)}