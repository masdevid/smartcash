"""
File: smartcash/ui/model/backbone/operations/build_operation.py
Description: Build operation handler for backbone architecture building
"""

from typing import Dict, Any, Callable, Optional
from smartcash.ui.core.handlers.operation_handler import OperationHandler, OperationResult
from ..services.backbone_service import BackboneService
from ..constants import BackboneOperation


class BuildOperation(OperationHandler):
    """Operation handler for backbone architecture building."""
    
    def __init__(self, operation_container=None):
        super().__init__(
            module_name='backbone_build',
            parent_module='model',
            operation_container=operation_container
        )
        self.backbone_service = BackboneService()
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations."""
        return {
            'build': self.build_architecture
        }
    
    async def build_architecture(self, 
                               config: Dict[str, Any],
                               progress_callback: Optional[Callable] = None,
                               log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Build backbone architecture.
        
        Args:
            config: Backbone configuration
            progress_callback: Optional callback for progress updates
            log_callback: Optional callback for logging
            
        Returns:
            Dict containing build results
        """
        try:
            self.log("Starting backbone architecture building", 'info')
            
            # Use the backbone service for building
            result = await self.backbone_service.build_backbone_architecture(
                config=config,
                progress_callback=progress_callback or self._create_progress_callback(),
                log_callback=log_callback or self._create_log_callback()
            )
            
            if result['success']:
                self.log("✅ Backbone architecture built successfully", 'success')
                
                # Log architecture information
                if 'stats' in result:
                    stats = result['stats']
                    self.log(f"📊 Architecture stats: {stats['total_parameters']:,} parameters, "
                           f"{stats['model_size_mb']:.1f} MB", 'info')
                
                if 'layer_info' in result:
                    layer_info = result['layer_info']
                    self.log(f"🏗️ Layer structure: {layer_info['total_layers']} total layers "
                           f"({layer_info['backbone_layers']} backbone + {layer_info['head_layers']} head)", 'info')
            else:
                self.log(f"❌ Building failed: {result.get('error', 'Unknown error')}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Build operation failed: {e}")
            self.log(f"❌ Build operation failed: {str(e)}", 'error')
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
        """Initialize the build operation handler."""
        return {
            'success': True,
            'operations': list(self.get_operations().keys())
        }