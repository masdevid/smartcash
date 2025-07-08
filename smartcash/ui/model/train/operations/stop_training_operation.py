"""
File: smartcash/ui/model/train/operations/stop_training_operation.py
Stop training operation handler.
"""

import asyncio
from typing import Dict, Any, Callable

from smartcash.ui.core.handlers.operation_handler import OperationHandler
from ..constants import TrainingOperation, OPERATION_PROGRESS_STEPS
from ..services.training_service import TrainingService


class StopTrainingOperation(OperationHandler):
    """Operation handler for stopping training."""
    
    def __init__(self, training_service: TrainingService = None):
        """Initialize the stop training operation."""
        super().__init__(module_name="train", parent_module="model")
        self.operation_type = TrainingOperation.STOP.value
        self.progress_steps = OPERATION_PROGRESS_STEPS[self.operation_type]
        self.training_service = training_service or TrainingService()
    
    def initialize(self):
        """Initialize the operation handler."""
        # Initialize any required resources
        pass
    
    def get_operations(self) -> Dict[str, Callable]:
        """
        Get available operations.
        
        Returns:
            Dictionary mapping operation names to callable methods
        """
        return {
            'stop_training': self.execute_operation
        }
    
    async def execute_operation(self, progress_callback: Callable = None,
                              log_callback: Callable = None) -> Dict[str, Any]:
        """
        Execute stop training operation.
        
        Args:
            progress_callback: Progress update callback
            log_callback: Logging callback
            
        Returns:
            Operation result dictionary
        """
        try:
            operation_id = f"stop_training_{asyncio.get_event_loop().time()}"
            
            if log_callback:
                log_callback("🛑 Stopping training operation...")
            
            # Step 1: Stop current training
            if progress_callback:
                progress_callback(20, self.progress_steps[0])
            
            stop_result = await self.training_service.stop_training(
                progress_callback=self._create_substep_callback(progress_callback, 20, 90),
                log_callback=log_callback
            )
            
            if progress_callback:
                progress_callback(100, self.progress_steps[2])
            
            if log_callback:
                if stop_result.get("success", False):
                    log_callback("✅ Training stopped successfully")
                else:
                    log_callback(f"⚠️ Stop training: {stop_result.get('message', 'Unknown error')}")
            
            return {
                "success": stop_result.get("success", False),
                "message": stop_result.get("message", "Training stop operation completed"),
                "operation_id": operation_id,
                "stop_result": stop_result,
                "phase": stop_result.get("phase", "unknown")
            }
            
        except Exception as e:
            error_msg = f"Stop training operation failed: {str(e)}"
            if log_callback:
                log_callback(f"❌ {error_msg}")
            
            return {
                "success": False,
                "message": error_msg,
                "operation_id": operation_id,
                "error": str(e)
            }
    
    def _create_substep_callback(self, main_callback: Callable, start_percent: int, end_percent: int) -> Callable:
        """Create a substep progress callback that maps to a portion of the main progress."""
        def substep_callback(percent: int, message: str = ""):
            if main_callback:
                # Map substep progress to main progress range
                mapped_percent = start_percent + (percent * (end_percent - start_percent) // 100)
                main_callback(mapped_percent, message)
        
        return substep_callback