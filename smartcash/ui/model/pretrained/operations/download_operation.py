"""
File: smartcash/ui/model/pretrained/operations/download_operation.py
Download operation handler for pretrained models.
"""

import asyncio
from typing import Dict, Any, Optional, Callable

from smartcash.ui.core.handlers.operation_handler import OperationHandler
from ..services.pretrained_service import PretrainedService
from ..constants import PretrainedOperation, PROGRESS_STEPS


class DownloadOperation(OperationHandler):
    """
    Download operation handler for pretrained models.
    Downloads YOLOv5s and EfficientNet-B4 models to specified directory.
    """
    
    def __init__(self):
        """Initialize the download operation handler."""
        super().__init__(module_name="pretrained", parent_module="model")
        self.service = PretrainedService()
        self.operation_type = PretrainedOperation.DOWNLOAD.value
        self.progress_steps = PROGRESS_STEPS[self.operation_type]
    
    def initialize(self) -> None:
        """Initialize the operation handler (required by base class)."""
        pass
    
    def get_operations(self) -> Dict[str, Callable]:
        """
        Get available operations (required by base class).
        
        Returns:
            Dictionary containing available operations
        """
        return {
            'download': self.execute_operation
        }
    
    async def execute_operation(self, config: Dict[str, Any], ui_components: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the pretrained models download operation.
        
        Args:
            config: Configuration containing models directory and URLs
            ui_components: UI components for progress tracking and logging
            **kwargs: Additional operation parameters
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Get progress and log callbacks
            progress_callback = self._get_progress_callback(ui_components)
            log_callback = self._get_log_callback(ui_components)
            
            # Start operation
            if log_callback:
                log_callback("🚀 Starting pretrained models download operation...")
            
            if progress_callback:
                progress_callback(5, self.progress_steps[0])  # "Checking existing models"
            
            # Check existing models first
            models_dir = config.get("models_dir", "/data/pretrained")
            existing_check = await self.service.check_existing_models(
                models_dir, progress_callback, log_callback
            )
            
            if progress_callback:
                progress_callback(15, self.progress_steps[1])  # "Preparing download directory"
            
            # Download all models
            download_results = await self.service.download_all_models(
                config, progress_callback, log_callback
            )
            
            # Prepare final results
            results = {
                "operation": self.operation_type,
                "success": download_results.get("all_successful", False),
                "models_dir": models_dir,
                "existing_models": existing_check,
                "download_results": download_results,
                "summary": self.service.get_models_summary(models_dir)
            }
            
            if progress_callback:
                progress_callback(100, self.progress_steps[-1])  # "Download complete"
            
            if results["success"]:
                if log_callback:
                    log_callback("✅ Pretrained models download operation completed successfully!")
            else:
                if log_callback:
                    success_count = download_results.get("success_count", 0)
                    total_count = download_results.get("total_count", 2)
                    log_callback(f"⚠️ Download operation completed with issues: {success_count}/{total_count} models downloaded")
            
            return results
            
        except Exception as e:
            error_msg = f"Download operation failed: {str(e)}"
            if log_callback:
                log_callback(f"❌ {error_msg}")
            
            return {
                "operation": self.operation_type,
                "success": False,
                "error": error_msg,
                "models_dir": config.get("models_dir", "/data/pretrained")
            }
    
    def _get_progress_callback(self, ui_components: Dict[str, Any]) -> Optional[callable]:
        """Get progress callback from UI components."""
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'update_progress'):
            return progress_tracker.update_progress
        return None
    
    def _get_log_callback(self, ui_components: Dict[str, Any]) -> Optional[callable]:
        """Get log callback from UI components."""
        log_output = ui_components.get('log_output')
        if log_output and hasattr(log_output, 'log'):
            return log_output.log
        return None