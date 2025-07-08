"""
File: smartcash/ui/model/backbone/operations/summary_operation.py
Description: Summary operation handler for backbone model summary generation
"""

from typing import Dict, Any, Callable, Optional
from smartcash.ui.core.handlers.operation_handler import OperationHandler, OperationResult
from ..services.backbone_service import BackboneService
from ..constants import BackboneOperation


class SummaryOperation(OperationHandler):
    """Operation handler for backbone model summary generation."""
    
    def __init__(self, operation_container=None):
        super().__init__(
            module_name='backbone_summary',
            parent_module='model',
            operation_container=operation_container
        )
        self.backbone_service = BackboneService()
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations."""
        return {
            'summary': self.generate_summary
        }
    
    async def generate_summary(self, 
                             config: Dict[str, Any],
                             progress_callback: Optional[Callable] = None,
                             log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Generate backbone model summary.
        
        Args:
            config: Backbone configuration
            progress_callback: Optional callback for progress updates
            log_callback: Optional callback for logging
            
        Returns:
            Dict containing summary results
        """
        try:
            self.log("Starting backbone model summary generation", 'info')
            
            # Use the backbone service for summary generation
            result = await self.backbone_service.generate_model_summary(
                config=config,
                progress_callback=progress_callback or self._create_progress_callback(),
                log_callback=log_callback or self._create_log_callback()
            )
            
            if result['success']:
                self.log("✅ Backbone model summary generated successfully", 'success')
                
                # Log summary information
                if 'summary' in result:
                    summary = result['summary']
                    backbone_type = summary.get('backbone_type', 'Unknown')
                    self.log(f"📋 Model summary: {backbone_type} backbone configured", 'info')
                
                if 'analysis' in result:
                    analysis = result['analysis']
                    self.log(f"📈 Performance analysis: {analysis.get('inference_speed', 'Unknown')} speed, "
                           f"{analysis.get('accuracy', 'Unknown')} accuracy", 'info')
            else:
                self.log(f"❌ Summary generation failed: {result.get('error', 'Unknown error')}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Summary operation failed: {e}")
            self.log(f"❌ Summary operation failed: {str(e)}", 'error')
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
        """Initialize the summary operation handler."""
        return {
            'success': True,
            'operations': list(self.get_operations().keys())
        }