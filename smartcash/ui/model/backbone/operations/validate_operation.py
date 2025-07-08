"""
File: smartcash/ui/model/backbone/operations/validate_operation.py
Description: Validate operation handler for backbone configuration
"""

from typing import Dict, Any, Callable, Optional
from smartcash.ui.core.handlers.operation_handler import OperationHandler, OperationResult
from ..services.backbone_service import BackboneService
from ..constants import BackboneOperation


class ValidateOperation(OperationHandler):
    """Operation handler for backbone configuration validation."""
    
    def __init__(self, operation_container=None):
        super().__init__(
            module_name='backbone_validate',
            parent_module='model',
            operation_container=operation_container
        )
        self.backbone_service = BackboneService()
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations."""
        return {
            'validate': self.validate_config
        }
    
    async def validate_config(self, 
                            config: Dict[str, Any],
                            progress_callback: Optional[Callable] = None,
                            log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Validate backbone configuration.
        
        Args:
            config: Backbone configuration to validate
            progress_callback: Optional callback for progress updates
            log_callback: Optional callback for logging
            
        Returns:
            Dict containing validation results
        """
        try:
            self.log("Starting backbone configuration validation", 'info')
            
            # Use the backbone service for validation
            result = await self.backbone_service.validate_backbone_config(
                config=config,
                progress_callback=progress_callback or self._create_progress_callback(),
                log_callback=log_callback or self._create_log_callback()
            )
            
            if result['valid']:
                self.log("✅ Backbone configuration validation completed successfully", 'success')
            else:
                self.log(f"❌ Validation failed: {', '.join(result['errors'])}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Validation operation failed: {e}")
            self.log(f"❌ Validation operation failed: {str(e)}", 'error')
            return {
                'valid': False,
                'error': str(e),
                'errors': [f"Operation failed: {str(e)}"],
                'warnings': []
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
        """Initialize the validate operation handler."""
        return {
            'success': True,
            'operations': list(self.get_operations().keys())
        }