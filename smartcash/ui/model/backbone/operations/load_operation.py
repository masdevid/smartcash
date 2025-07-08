"""
File: smartcash/ui/model/backbone/operations/load_operation.py
Description: Load operation handler for backbone model loading
"""

from typing import Dict, Any, Callable, Optional
from smartcash.ui.core.handlers.operation_handler import OperationHandler, OperationResult
from ..services.backbone_service import BackboneService
from ..constants import BackboneOperation


class LoadOperation(OperationHandler):
    """Operation handler for backbone model loading."""
    
    def __init__(self, operation_container=None):
        super().__init__(
            module_name='backbone_load',
            parent_module='model',
            operation_container=operation_container
        )
        self.backbone_service = BackboneService()
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations."""
        return {
            'load': self.load_model
        }
    
    async def load_model(self, 
                        config: Dict[str, Any],
                        progress_callback: Optional[Callable] = None,
                        log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Load backbone model.
        
        Args:
            config: Backbone configuration
            progress_callback: Optional callback for progress updates
            log_callback: Optional callback for logging
            
        Returns:
            Dict containing load results
        """
        try:
            self.log("Starting backbone model loading", 'info')
            
            # Use the backbone service for loading
            result = await self.backbone_service.load_backbone_model(
                config=config,
                progress_callback=progress_callback or self._create_progress_callback(),
                log_callback=log_callback or self._create_log_callback()
            )
            
            if result['success']:
                self.log("✅ Backbone model loaded successfully", 'success')
                
                # Log model information
                if 'info' in result:
                    info = result['info']
                    self.log(f"📊 Model info: {info['total_parameters']:,} parameters, "
                           f"{info['model_size_mb']:.1f} MB", 'info')
            else:
                self.log(f"❌ Loading failed: {result.get('error', 'Unknown error')}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Load operation failed: {e}")
            self.log(f"❌ Load operation failed: {str(e)}", 'error')
            return {
                'success': False,
                'error': str(e),
                'message': f"Operation failed: {str(e)}"
            }
    
    def _create_progress_callback(self) -> Callable:
        """Create a progress callback function."""
        async def callback(current: int, total: int, message: str):
            if hasattr(self, '_operation_container') and self._operation_container:
                progress_percent = int((current / total) * 100) if total > 0 else 0
                self._operation_container.update_progress(
                    value=progress_percent,
                    message=message,
                    level='primary'
                )
        return callback
    
    def _create_log_callback(self) -> Callable:
        """Create a log callback function."""
        async def callback(level: str, message: str):
            self.log(message, level.lower())
        return callback
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize the load operation handler."""
        return {
            'success': True,
            'operations': list(self.get_operations().keys())
        }